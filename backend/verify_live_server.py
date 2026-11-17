"""
Live HTTP Verification Script for RoadVision AI Backend Server
Pings GET /health and tests POST /predict with a real damaged road image.
Uses httpx (installed in virtual environment).
"""
import sys
import time
from pathlib import Path
import httpx

BASE_URL = "http://127.0.0.1:8000"

def test_live_server():
    print(f"Connecting to live server at {BASE_URL}...")
    
    # 1. Health check with retries for server startup
    max_retries = 15
    client = httpx.Client(timeout=30.0)
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.get(f"{BASE_URL}/health")
            if resp.status_code == 200:
                print(f"[SUCCESS] Server is online (attempt {attempt})!")
                health_data = resp.json()
                print("Health Check Response:")
                print(health_data)
                assert health_data["status"] == "healthy", "Server status not healthy"
                assert health_data["model_loaded"] is True, "Model not loaded"
                break
        except httpx.RequestError:
            print(f"Waiting for server... ({attempt}/{max_retries})")
            time.sleep(2)
    else:
        print("[ERROR] Timed out waiting for server.")
        sys.exit(1)

    # 2. Test POST /predict with real road image
    sample_img = Path(__file__).resolve().parent / "samples" / "road_pothole_and_crack.jpg"
    assert sample_img.exists(), f"Sample image {sample_img} does not exist"
    
    print(f"\nTesting POST /predict with real image: {sample_img.name}...")
    with open(sample_img, "rb") as f:
        files = {"file": (sample_img.name, f, "image/jpeg")}
        resp = client.post(f"{BASE_URL}/predict", files=files)
    
    assert resp.status_code == 200, f"Predict failed with status {resp.status_code}: {resp.text}"
    result = resp.json()

    print("[SUCCESS] POST /predict succeeded!")
    print("=" * 60)
    print(f"Filename: {result.get('filename')}")
    print(f"Model: {result.get('model')}")
    print(f"Dimensions: {result.get('image_dimensions')}")
    print(f"Pothole Area: {result.get('pothole_area_percentage')}%")
    print(f"Crack Area: {result.get('crack_area_percentage')}%")
    print(f"Total Damaged Area: {result.get('total_damaged_area_percentage')}%")
    print(f"Severity: {result.get('severity')} - {result.get('severity_description')}")
    print(f"Overlay Image URL: {result.get('overlay_image')[:50]}... (total {len(result.get('overlay_image'))} chars)")
    print(f"Mask Image URL: {result.get('mask_image')[:50]}... (total {len(result.get('mask_image'))} chars)")
    print("=" * 60)

    # Verify predictions are non-zero for damaged road
    assert result.get('total_damaged_area_percentage', 0) > 0, "Damage percentage unexpectedly zero"
    print("\n[VERIFICATION PASSED] All endpoints verified and confirmed functional!")

if __name__ == "__main__":
    test_live_server()
