"""
RoadVision AI - Semantic Segmentation FastAPI Backend
Serves inference for DeepLabV3-ResNet50 road damage segmentation model
(Classes: Background, Pothole, Crack)
"""

import io
import os
import json
import base64
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import (
    init_db,
    create_inspection,
    get_inspections,
    get_inspection_by_id,
    delete_inspection,
    get_analytics_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("roadvision-backend")

# Define base paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ARTIFACTS_DIR_CANDIDATES = [
    PROJECT_ROOT / "artifacts" / "roadvision_model_artifacts",
    BASE_DIR / "artifacts" / "roadvision_model_artifacts",
    Path("artifacts/roadvision_model_artifacts"),
    Path("../artifacts/roadvision_model_artifacts"),
]


def resolve_artifacts_path() -> Path:
    """Find the directory containing model artifacts."""
    for candidate in ARTIFACTS_DIR_CANDIDATES:
        if candidate.exists() and (candidate / "best_model.pth").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate model artifacts directory containing 'best_model.pth'. "
        f"Searched: {[str(c) for c in ARTIFACTS_DIR_CANDIDATES]}"
    )


def extract_bounding_boxes(
    pred_mask: np.ndarray,
    probs_512: np.ndarray,
    orig_w: int,
    orig_h: int,
    min_area_pothole: int = 40,
    min_area_crack: int = 25
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts bounding boxes strictly from the predicted segmentation mask using
    8-connected component contour analysis. Pothole and Crack classes are evaluated independently.
    
    Coordinates are scaled back to the original image dimensions without distortion.
    Noise filtering removes isolated components smaller than the configurable min_area threshold.
    Confidence is derived from genuine mean softmax class probability over the segmented pixels.
    """
    boxes = []
    
    # 1. Resize class probability maps to match original image dimensions for accurate pixel-level confidence
    prob_pothole = np.array(Image.fromarray(probs_512[1]).resize((orig_w, orig_h), Image.Resampling.BILINEAR))
    prob_crack = np.array(Image.fromarray(probs_512[2]).resize((orig_w, orig_h), Image.Resampling.BILINEAR))

    class_configs = [
        {
            "class_id": 1,
            "class_name": "Pothole",
            "min_area": min_area_pothole,
            "prob_map": prob_pothole,
            "color_rgb": (255, 59, 48),    # Red
            "color_hex": "#FF3B30",
        },
        {
            "class_id": 2,
            "class_name": "Crack",
            "min_area": min_area_crack,
            "prob_map": prob_crack,
            "color_rgb": (255, 204, 0),   # Yellow
            "color_hex": "#FFCC00",
        }
    ]

    pothole_count = 0
    crack_count = 0

    for cfg in class_configs:
        cid = cfg["class_id"]
        cname = cfg["class_name"]
        binary_mask = (pred_mask == cid).astype(np.uint8)

        # Skip if no pixels predicted for this class
        if not np.any(binary_mask):
            continue

        # 8-connectivity connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        region_idx = 0
        for k in range(1, num_labels):
            x = int(stats[k, cv2.CC_STAT_LEFT])
            y = int(stats[k, cv2.CC_STAT_TOP])
            w = int(stats[k, cv2.CC_STAT_WIDTH])
            h = int(stats[k, cv2.CC_STAT_HEIGHT])
            area = int(stats[k, cv2.CC_STAT_AREA])

            # Filter tiny noisy regions based on configurable minimum pixel area
            if area < cfg["min_area"]:
                continue

            region_idx += 1
            if cid == 1:
                pothole_count += 1
            else:
                crack_count += 1

            component_mask = (labels == k)
            region_probs = cfg["prob_map"][component_mask]
            mean_conf = float(np.mean(region_probs)) if len(region_probs) > 0 else 0.0

            boxes.append({
                "id": f"{cname.lower()}_{region_idx}",
                "class_id": cid,
                "class_name": cname,
                "box": [x, y, w, h],
                "coordinates": {
                    "xmin": x,
                    "ymin": y,
                    "xmax": x + w,
                    "ymax": y + h,
                    "width": w,
                    "height": h
                },
                "area_pixels": area,
                "bbox_area": w * h,
                "confidence": round(mean_conf, 4),
                "confidence_percentage": round(mean_conf * 100.0, 1),
                "confidence_definition": "Mean softmax probability of the predicted class across all segmented pixels within this connected component.",
                "color_rgb": list(cfg["color_rgb"]),
                "color_hex": cfg["color_hex"]
            })

    damage_stats = {
        "total_pothole_regions": pothole_count,
        "total_crack_regions": crack_count,
        "total_damage_regions": len(boxes),
        "has_damage": len(boxes) > 0,
        "status_message": "Damage detected" if len(boxes) > 0 else "No damage detected",
        "min_area_thresholds": {
            "pothole": min_area_pothole,
            "crack": min_area_crack
        }
    }

    return boxes, damage_stats


def render_bounding_boxes(
    orig_np: np.ndarray,
    boxes: List[Dict[str, Any]],
    line_thickness: int = 2
) -> np.ndarray:
    """
    Renders rectangular bounding boxes and filled text badges onto an RGB image.
    Uses red outlines for potholes and yellow outlines for road cracks.
    """
    if len(boxes) == 0:
        return orig_np.copy()

    vis_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    img_h, img_w = orig_np.shape[:2]

    # Dynamically scale font and thickness with image resolution
    scale = max(img_w, img_h) / 800.0
    font_scale = max(0.40, 0.45 * scale)
    thickness = max(1, int(line_thickness * scale))
    badge_pad = max(2, int(4 * scale))

    for b in boxes:
        cid = b["class_id"]
        cname = b["class_name"]
        x, y, w, h = b["box"]
        conf_pct = b.get("confidence_percentage", 0.0)
        label_text = f"{cname} {conf_pct:.1f}%"

        # Colors in BGR
        if cid == 1:  # Pothole: Red (#FF3B30)
            box_bgr = (48, 59, 255)
            badge_bg = (48, 59, 255)
            text_color = (255, 255, 255)
        else:         # Crack: Yellow (#FFCC00)
            box_bgr = (0, 204, 255)
            badge_bg = (0, 204, 255)
            text_color = (23, 36, 53)  # Dark slate for contrast on yellow

        # 1. Draw Bounding Rectangle
        cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), box_bgr, thickness=thickness)

        # 2. Compute Text Badge Dimensions
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=max(1, int(thickness * 0.75))
        )
        badge_w = text_w + 2 * badge_pad
        badge_h = text_h + 2 * badge_pad + baseline

        # Determine badge placement (above box if space permits, else inside)
        if y - badge_h >= 0:
            badge_y1 = y - badge_h
            badge_y2 = y
            text_y = y - badge_pad - baseline
        else:
            badge_y1 = y
            badge_y2 = y + badge_h
            text_y = y + badge_pad + text_h

        badge_x1 = max(0, x)
        badge_x2 = min(img_w, badge_x1 + badge_w)

        # Draw filled badge background
        cv2.rectangle(vis_bgr, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_bg, -1)
        # Draw subtle badge border
        cv2.rectangle(vis_bgr, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 0, 0), 1)

        # Draw text inside badge
        cv2.putText(
            vis_bgr,
            label_text,
            (badge_x1 + badge_pad, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness=max(1, int(thickness * 0.75)),
            lineType=cv2.LINE_AA
        )

    return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)


class ModelManager:
    """Singleton model container loaded once at application startup."""

    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.config: Dict[str, Any] = {}
        self.class_mapping: Dict[str, Any] = {}
        self.artifacts_dir: Optional[Path] = None
        self.is_loaded: bool = False

    def load(self):
        """Loads configuration, class mappings, and model weights into memory."""
        self.artifacts_dir = resolve_artifacts_path()
        logger.info(f"Loading artifacts from: {self.artifacts_dir}")

        # 1. Load config
        config_path = self.artifacts_dir / "model_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # 2. Load class mapping
        mapping_path = self.artifacts_dir / "class_mapping.json"
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.class_mapping = json.load(f)

        num_classes = self.config.get("num_classes", 3)
        aux_loss = self.config.get("aux_loss", True)

        logger.info(
            f"Building architecture: {self.config.get('architecture', 'DeepLabV3_ResNet50')} "
            f"with num_classes={num_classes}, aux_loss={aux_loss} on device={self.device}"
        )

        # 3. Instantiate model matching training configuration
        # Note: weights_backbone=None prevents downloading backbone weights since
        # best_model.pth already contains all weights (backbone, classifier, aux).
        model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            aux_loss=aux_loss
        )

        weights_path = self.artifacts_dir / "best_model.pth"
        logger.info(f"Loading model weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        self.model = model
        self.is_loaded = True
        logger.info("RoadVision segmentation model successfully loaded and ready for inference.")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Executes end-to-end inference on a PIL RGB image.
        Returns segmentation mask, blended overlay, and class metrics.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Cannot run inference.")

        orig_w, orig_h = image.size
        target_res = self.config.get("input_resolution", {"height": 512, "width": 512})
        target_h = target_res.get("height", 512)
        target_w = target_res.get("width", 512)

        # 1. Preprocess image
        resized_img = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
        img_np = np.array(resized_img, dtype=np.float32) / 255.0

        norm_cfg = self.config.get("normalization", {})
        mean = np.array(norm_cfg.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(norm_cfg.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        norm_img = (img_np - mean) / std

        # Convert to tensor: (B, C, H, W)
        tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # 2. Run inference
        with torch.no_grad():
            output = self.model(tensor)
            logits = output["out"]  # shape: (1, 3, 512, 512)
            probs_512 = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_512 = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # 3. Postprocess mask: Resize back to original image resolution
        mask_pil = Image.fromarray(pred_512).resize((orig_w, orig_h), Image.Resampling.NEAREST)
        pred_mask = np.array(mask_pil, dtype=np.uint8)

        # 4. Extract mask-derived bounding boxes using connected component analysis
        bounding_boxes, damage_stats = extract_bounding_boxes(
            pred_mask=pred_mask,
            probs_512=probs_512,
            orig_w=orig_w,
            orig_h=orig_h,
            min_area_pothole=40,
            min_area_crack=25
        )

        # 5. Generate colored segmentation mask
        # Colors: Background -> [0, 0, 0], Pothole -> [255, 59, 48], Crack -> [255, 204, 0]
        color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        classes_cfg = self.class_mapping.get("classes", {})
        
        pothole_rgb = classes_cfg.get("1", {}).get("color_rgb", [255, 59, 48])
        crack_rgb = classes_cfg.get("2", {}).get("color_rgb", [255, 204, 0])

        color_mask[pred_mask == 1] = pothole_rgb
        color_mask[pred_mask == 2] = crack_rgb

        # 6. Generate alpha-blended overlay on original image
        orig_np = np.array(image.convert("RGB"), dtype=np.uint8)
        overlay_np = orig_np.copy()
        damage_mask = (pred_mask > 0)
        alpha = 0.50
        if np.any(damage_mask):
            blended = (orig_np[damage_mask] * (1.0 - alpha) + color_mask[damage_mask] * alpha)
            overlay_np[damage_mask] = np.clip(blended, 0, 255).astype(np.uint8)

        # 7. Generate mask-derived bounding box visualization image
        bbox_vis_np = render_bounding_boxes(orig_np, bounding_boxes)

        # 8. Calculate pixel areas and percentages
        total_pixels = int(orig_w * orig_h)
        background_pixels = int(np.sum(pred_mask == 0))
        pothole_pixels = int(np.sum(pred_mask == 1))
        crack_pixels = int(np.sum(pred_mask == 2))
        total_damaged_pixels = pothole_pixels + crack_pixels

        background_pct = round((background_pixels / total_pixels) * 100.0, 2)
        pothole_pct = round((pothole_pixels / total_pixels) * 100.0, 2)
        crack_pct = round((crack_pixels / total_pixels) * 100.0, 2)
        total_damaged_pct = round((total_damaged_pixels / total_pixels) * 100.0, 2)

        # Attach percentage statistics to damage_stats
        damage_stats["pothole_area_percentage"] = pothole_pct
        damage_stats["crack_area_percentage"] = crack_pct
        damage_stats["total_damaged_area_percentage"] = total_damaged_pct
        damage_stats["detected_regions"] = bounding_boxes

        # Determine road condition severity
        if total_damaged_pct == 0.0:
            severity = "Good"
            severity_description = "No pavement damage detected."
        elif total_damaged_pct < 1.5:
            severity = "Low"
            severity_description = "Minor surface damage detected."
        elif total_damaged_pct < 5.0:
            severity = "Moderate"
            severity_description = "Noticeable damage requiring planned maintenance."
        else:
            severity = "High / Critical"
            severity_description = "Significant pavement degradation requiring prompt repair."

        # 9. Convert visualization images to Base64
        mask_b64 = self._image_to_base64(Image.fromarray(color_mask), format="PNG")
        overlay_b64 = self._image_to_base64(Image.fromarray(overlay_np), format="JPEG")
        bbox_b64 = self._image_to_base64(Image.fromarray(bbox_vis_np), format="JPEG")
        orig_b64 = self._image_to_base64(image.convert("RGB"), format="JPEG")

        return {
            "status": "success",
            "model": self.config.get("model_name", "RoadVision_DeepLabV3_ResNet50"),
            "image_dimensions": {
                "width": orig_w,
                "height": orig_h
            },
            "pothole_area_percentage": pothole_pct,
            "crack_area_percentage": crack_pct,
            "total_damaged_area_percentage": total_damaged_pct,
            "background_area_percentage": background_pct,
            "severity": severity,
            "severity_description": severity_description,
            "metrics": {
                "pothole_area_percentage": pothole_pct,
                "crack_area_percentage": crack_pct,
                "total_damaged_area_percentage": total_damaged_pct,
                "background_area_percentage": background_pct,
                "pixels": {
                    "background": background_pixels,
                    "pothole": pothole_pixels,
                    "crack": crack_pixels,
                    "total": total_pixels
                }
            },
            "bounding_boxes": bounding_boxes,
            "boxes": bounding_boxes,
            "damage_statistics": damage_stats,
            "class_labels": {
                "0": classes_cfg.get("0", {}).get("label", "Road Surface / Background"),
                "1": classes_cfg.get("1", {}).get("label", "Pothole"),
                "2": classes_cfg.get("2", {}).get("label", "Road Crack")
            },
            "images": {
                "original": f"data:image/jpeg;base64,{orig_b64}",
                "overlay": f"data:image/jpeg;base64,{overlay_b64}",
                "mask": f"data:image/png;base64,{mask_b64}",
                "bbox": f"data:image/jpeg;base64,{bbox_b64}"
            },
            # Flat convenience fields for frontend components
            "original_image": f"data:image/jpeg;base64,{orig_b64}",
            "overlay_image": f"data:image/jpeg;base64,{overlay_b64}",
            "mask_image": f"data:image/png;base64,{mask_b64}",
            "bbox_image": f"data:image/jpeg;base64,{bbox_b64}"
        }

    @staticmethod
    def _image_to_base64(img: Image.Image, format: str = "PNG") -> str:
        buffer = io.BytesIO()
        img.save(buffer, format=format, quality=90 if format == "JPEG" else None)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


model_manager = ModelManager()


def seed_initial_inspections():
    """Seeds database with verified real model predictions if empty or updates missing bounding box data."""
    existing = get_inspections()
    samples_dir = BASE_DIR / "samples"
    if not samples_dir.exists():
        return

    # If already seeded and has bbox_image, no need to re-seed
    if len(existing) > 0 and any(item.get("bbox_image") for item in existing):
        return

    logger.info("Seeding initial real inspections from verified road samples with bounding boxes...")
    sample_files = list(samples_dir.glob("*.jpg"))
    sample_titles = {
        "road_pothole_and_crack.jpg": "Sector 4 Highway - Pothole & Longitudinal Crack Survey",
        "road_multiple_damage.jpg": "Ring Road Arterial - Multi-damage Surface Assessment",
        "road_cracks.jpg": "West Avenue Corridor - Alligator & Transverse Crack Review",
        "road_potholes.jpg": "Industrial Bypass - Deep Pothole Clustered Damage",
    }

    for sample_path in sample_files:
        try:
            img = Image.open(sample_path)
            res = model_manager.predict(img)
            res["filename"] = sample_path.name
            res["title"] = sample_titles.get(sample_path.name, sample_path.stem.replace("_", " ").title())
            res["notes"] = f"Automated inspection conducted using DeepLabV3-ResNet50 model with mask-derived bounding box analysis. Condition assessed as {res['severity']}."
            create_inspection(res)
            logger.info(f"Seeded real inspection record: {res['title']}")
        except Exception as e:
            logger.warning(f"Could not seed sample {sample_path.name}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and loads the model exactly once upon backend startup."""
    logger.info("Initializing RoadVision AI Backend...")
    init_db()
    model_manager.load()
    seed_initial_inspections()
    yield
    logger.info("Shutting down RoadVision AI Backend...")


app = FastAPI(
    title="RoadVision AI API",
    description="DeepLabV3-ResNet50 Semantic Segmentation API for Potholes and Cracks",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for Vite frontend and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Health check endpoint reporting model status and device."""
    return {
        "status": "healthy",
        "service": "RoadVision AI Segmentation Backend",
        "version": "1.0.0",
        "model_loaded": model_manager.is_loaded,
        "model_name": model_manager.config.get("model_name", "RoadVision_DeepLabV3_ResNet50"),
        "architecture": model_manager.config.get("architecture", "DeepLabV3_ResNet50"),
        "device": model_manager.device,
        "classes": {
            "0": "background",
            "1": "pothole",
            "2": "crack"
        }
    }


@app.post("/predict")
@app.post("/api/predict")
async def predict_road_damage(file: UploadFile = File(...)):
    """
    Image-upload prediction endpoint.
    Accepts an image file, runs DeepLabV3-ResNet50 inference,
    and returns pixel-level damage quantification, colored mask, and overlay.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently loading or unavailable."
        )

    # Validate content type if provided
    if file.content_type and not (
        file.content_type.startswith("image/") or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Please upload an image (JPEG, PNG, WEBP)."
        )

    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as img_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot decode image file: {str(img_err)}"
            )

        # Run inference
        results = model_manager.predict(image)
        results["filename"] = file.filename
        return JSONResponse(content=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Inference error occurred:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


# ============================================================================
# Inspection & Analytics Database Endpoints
# ============================================================================

@app.get("/api/inspections")
async def list_inspections(search: Optional[str] = Query(None)):
    """Retrieves all saved inspections, optionally filtered by search term."""
    return get_inspections(search=search)


@app.post("/api/inspections")
async def save_inspection(payload: Dict[str, Any] = Body(...)):
    """Saves an inspection record into persistent storage."""
    try:
        saved = create_inspection(payload)
        return {"status": "success", "inspection": saved}
    except Exception as e:
        logger.exception("Failed to save inspection:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save inspection: {str(e)}"
        )


@app.get("/api/inspections/{insp_id}")
async def get_inspection(insp_id: str):
    """Retrieves full details and images for a specific inspection by ID."""
    insp = get_inspection_by_id(insp_id)
    if not insp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{insp_id}' not found."
        )
    return insp


@app.delete("/api/inspections/{insp_id}")
async def remove_inspection(insp_id: str):
    """Deletes an inspection record by ID."""
    success = delete_inspection(insp_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{insp_id}' not found."
        )
    return {"status": "success", "message": f"Inspection '{insp_id}' deleted successfully."}


@app.get("/api/analytics")
async def get_analytics():
    """Returns aggregated road inspection analytics calculated from saved data."""
    return get_analytics_summary()


@app.get("/api/samples")
async def get_sample_images():
    """Lists available test road images for one-click inspection in the UI."""
    samples_dir = BASE_DIR / "samples"
    if not samples_dir.exists():
        return []
    
    samples = []
    descriptions = {
        "road_pothole_and_crack.jpg": {"title": "Highway Sector 4", "desc": "Dual hazard: Deep pothole with longitudinal cracking."},
        "road_multiple_damage.jpg": {"title": "Ring Road Arterial", "desc": "Multiple surface distress areas and pavement fracture."},
        "road_cracks.jpg": {"title": "West Avenue Corridor", "desc": "Extensive branching alligator cracks on asphalt."},
        "road_potholes.jpg": {"title": "Industrial Bypass", "desc": "Severe cluster of potholes on weathered roadway."},
    }

    for p in samples_dir.glob("*.jpg"):
        meta = descriptions.get(p.name, {"title": p.stem.replace("_", " ").title(), "desc": "Real pavement sample."})
        samples.append({
            "filename": p.name,
            "title": meta["title"],
            "description": meta["desc"],
            "url": f"/api/samples/{p.name}"
        })
    return samples


@app.get("/api/samples/{filename}")
async def get_sample_file(filename: str):
    """Serves a sample image file."""
    sample_path = BASE_DIR / "samples" / filename
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    from fastapi.responses import FileResponse
    return FileResponse(sample_path, media_type="image/jpeg")


# ============================================================================
# Static Frontend SPA Mount
# ============================================================================
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    logger.info(f"Mounting compiled production frontend from {frontend_dist}")
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
