"""
Test suite for RoadVision AI FastAPI Backend
Validates model loading, inference, area percentages, and API endpoints.
"""

import io
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure backend directory is in python path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from main import app, ModelManager, resolve_artifacts_path, extract_bounding_boxes, render_bounding_boxes
from fastapi.testclient import TestClient


def test_artifacts_resolution():
    artifacts_dir = resolve_artifacts_path()
    assert artifacts_dir.exists(), f"Artifacts dir {artifacts_dir} does not exist"
    assert (artifacts_dir / "best_model.pth").exists(), "best_model.pth missing"
    assert (artifacts_dir / "model_config.json").exists(), "model_config.json missing"
    assert (artifacts_dir / "class_mapping.json").exists(), "class_mapping.json missing"
    print(f"[PASS] Artifacts resolved successfully: {artifacts_dir}")


def test_bounding_box_extraction_unit():
    """Unit tests for mask-derived bounding box extraction and noise filtering."""
    orig_w, orig_h = 600, 400
    pred_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    # Synthetic pothole at (100, 100) of size 50x40 (area 2000 px)
    pred_mask[100:140, 100:150] = 1
    # Synthetic tiny pothole noise of size 3x3 (area 9 px - should be filtered)
    pred_mask[50:53, 50:53] = 1

    # Synthetic crack at (300, 200) of size 100x10 (area 1000 px)
    pred_mask[200:210, 300:400] = 2
    # Synthetic tiny crack noise of size 2x2 (area 4 px - should be filtered)
    pred_mask[10:12, 10:12] = 2

    # Synthetic probability maps (shape 3, 512, 512)
    probs_512 = np.zeros((3, 512, 512), dtype=np.float32)
    probs_512[1, :, :] = 0.88  # 88% pothole prob
    probs_512[2, :, :] = 0.94  # 94% crack prob

    boxes, stats = extract_bounding_boxes(
        pred_mask=pred_mask,
        probs_512=probs_512,
        orig_w=orig_w,
        orig_h=orig_h,
        min_area_pothole=40,
        min_area_crack=25
    )

    # Validate counts (noise specks must be filtered out)
    assert len(boxes) == 2, f"Expected exactly 2 valid damage regions, got {len(boxes)}"
    assert stats["total_pothole_regions"] == 1
    assert stats["total_crack_regions"] == 1
    assert stats["has_damage"] is True

    # Validate pothole box
    p_box = [b for b in boxes if b["class_name"] == "Pothole"][0]
    assert p_box["box"] == [100, 100, 50, 40], f"Pothole box mismatch: {p_box['box']}"
    assert p_box["area_pixels"] == 2000
    assert p_box["confidence_percentage"] > 80.0

    # Validate crack box
    c_box = [b for b in boxes if b["class_name"] == "Crack"][0]
    assert c_box["box"] == [300, 200, 100, 10], f"Crack box mismatch: {c_box['box']}"
    assert c_box["area_pixels"] == 1000
    assert c_box["confidence_percentage"] > 90.0

    # Test empty mask produces 0 boxes and clean status
    empty_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    empty_boxes, empty_stats = extract_bounding_boxes(
        empty_mask, probs_512, orig_w, orig_h
    )
    assert len(empty_boxes) == 0, "Empty mask should produce 0 boxes"
    assert empty_stats["has_damage"] is False
    assert empty_stats["status_message"] == "No damage detected"

    # Test rendering bounding boxes
    orig_np = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    rendered = render_bounding_boxes(orig_np, boxes)
    assert rendered.shape == (orig_h, orig_w, 3)

    print("[PASS] Bounding-box connected component extraction & noise filtering passed.")


def test_model_manager_inference():
    manager = ModelManager()
    manager.load()
    assert manager.is_loaded, "Model should be loaded"
    assert manager.model is not None, "Model instance should exist"

    # Create synthetic road image (640x480 RGB)
    test_img = Image.new("RGB", (640, 480), color=(80, 80, 80))
    result = manager.predict(test_img)

    assert result["status"] == "success"
    assert "pothole_area_percentage" in result
    assert "crack_area_percentage" in result
    assert "total_damaged_area_percentage" in result
    assert "bounding_boxes" in result
    assert "damage_statistics" in result
    assert "images" in result
    assert result["images"]["overlay"].startswith("data:image/jpeg;base64,")
    assert result["images"]["mask"].startswith("data:image/png;base64,")
    assert result["images"]["bbox"].startswith("data:image/jpeg;base64,")
    assert result["bbox_image"].startswith("data:image/jpeg;base64,")
    assert result["image_dimensions"]["width"] == 640
    assert result["image_dimensions"]["height"] == 480
    print("[PASS] ModelManager inference passed with proper keys, bounding boxes, and Base64 outputs.")


def test_fastapi_endpoints():
    with TestClient(app) as client:
        # 1. Health check
        resp = client.get("/health")
        assert resp.status_code == 200, f"Health check failed: {resp.text}"
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        print(f"[PASS] GET /health returned 200 OK: {data['model_name']}")

        # 2. Prediction endpoint with synthetic image
        img = Image.new("RGB", (512, 512), color=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        files = {"file": ("test_road.jpg", buf, "image/jpeg")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 200, f"Predict failed: {resp.text}"
        pred_data = resp.json()
        assert pred_data["status"] == "success"
        assert "total_damaged_area_percentage" in pred_data
        assert "overlay_image" in pred_data
        assert "mask_image" in pred_data
        assert "bbox_image" in pred_data
        assert "bounding_boxes" in pred_data
        assert "damage_statistics" in pred_data
        print(f"[PASS] POST /predict returned 200 OK. Damage area: {pred_data['total_damaged_area_percentage']}%, Regions: {len(pred_data['bounding_boxes'])}")

        # 3. Prediction with real sample image if available
        sample_path = backend_dir / "samples" / "road_pothole_and_crack.jpg"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                files = {"file": ("road_pothole_and_crack.jpg", f, "image/jpeg")}
                resp = client.post("/predict", files=files)
                assert resp.status_code == 200, f"Predict on sample failed: {resp.text}"
                sample_data = resp.json()
                print(f"[PASS] POST /predict on sample image returned 200 OK:")
                print(f"       Pothole: {sample_data['pothole_area_percentage']}%, Crack: {sample_data['crack_area_percentage']}%, Total: {sample_data['total_damaged_area_percentage']}%")
                print(f"       Detected Bounding Boxes: {len(sample_data['bounding_boxes'])} regions (Potholes: {sample_data['damage_statistics']['total_pothole_regions']}, Cracks: {sample_data['damage_statistics']['total_crack_regions']})")
                print(f"       Severity: {sample_data['severity']}")


if __name__ == "__main__":
    print("Running RoadVision AI backend test suite...")
    test_artifacts_resolution()
    test_bounding_box_extraction_unit()
    test_model_manager_inference()
    test_fastapi_endpoints()
    print("All backend tests completed successfully!")
