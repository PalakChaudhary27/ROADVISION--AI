"""
Test RoadVision AI model directly on real dataset road images with potholes and cracks.
"""
from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import ModelManager

def main():
    print("Loading RoadVision model...")
    manager = ModelManager()
    manager.load()

    samples_dir = Path(__file__).resolve().parent / "samples"
    for sample in samples_dir.glob("*.jpg"):
        img = Image.open(sample)
        res = manager.predict(img)
        print("=" * 60)
        print(f"Sample: {sample.name}")
        print(f"Original Dimensions: {res['image_dimensions']['width']}x{res['image_dimensions']['height']}")
        print(f"Pothole Area: {res['pothole_area_percentage']}% ({res['metrics']['pixels']['pothole']} px)")
        print(f"Crack Area: {res['crack_area_percentage']}% ({res['metrics']['pixels']['crack']} px)")
        print(f"Total Damaged Area: {res['total_damaged_area_percentage']}%")
        print(f"Severity: {res['severity']} - {res['severity_description']}")
        print(f"Overlay data URL valid: {res['images']['overlay'].startswith('data:image/jpeg;base64,')}")
        print(f"Mask data URL valid: {res['images']['mask'].startswith('data:image/png;base64,')}")

if __name__ == "__main__":
    main()
