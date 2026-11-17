# 🛣️ RoadVision AI

**Automated pothole and road crack detection through pixel-level semantic segmentation**

RoadVision AI is an end-to-end road inspection platform that takes a standard RGB photograph of a road surface and returns a pixel-accurate map of surface damage — distinguishing potholes from cracks — along with area quantification, severity assessment, and mask-derived bounding boxes. The system pairs a fine-tuned **DeepLabV3 + ResNet-50** deep learning model with a production **FastAPI** backend and an interactive **React 18** web frontend.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Segmentation Classes](#segmentation-classes)
5. [Dataset](#dataset)
6. [Model Training](#model-training)
7. [Evaluation Results](#evaluation-results)
8. [Technology Stack](#technology-stack)
9. [Project Structure](#project-structure)
10. [Installation & Setup](#installation--setup)
11. [API Reference](#api-reference)
12. [Limitations & Future Work](#limitations--future-work)
13. [Author](#author)

---

## Problem Statement

Road surface degradation — primarily potholes and pavement cracks — poses serious risks to vehicle safety and imposes significant maintenance costs on municipal infrastructure. Traditional inspection methods rely on manual visual surveys that are slow, subjective, labour-intensive, and hazardous for field crews.

Standard object detection models (YOLO, SSD) produce only coarse bounding boxes that cannot capture the irregular, non-convex boundaries of real road damage or quantify affected surface area. RoadVision AI addresses this by applying **pixel-level semantic segmentation**, which delineates every damaged pixel and reports exact area percentages as a basis for maintenance prioritization.

---

## Key Features

| Feature | Detail |
|---|---|
| **Pixel-level segmentation** | Classifies every pixel into Background, Pothole, or Crack |
| **Mask-derived bounding boxes** | Extracted deterministically from the predicted mask using 8-connected component analysis — no secondary detector required |
| **Damage area quantification** | Reports pothole %, crack %, and total damage % as a fraction of the image frame |
| **Severity assessment** | Four-level road condition rating based on combined damage percentage |
| **Multi-mode visualization** | Distress Overlay, Pure Class Mask, Source Image, and Side-by-Side views |
| **Inspection history** | SQLite-backed record of all past inspections with search and analytics |
| **Built-in sample gallery** | Four bundled test road images for instant one-click demonstration |
| **Firebase authentication** | User login and signup integrated via Firebase Auth |

---

## System Architecture

```
[ User uploads road image via React UI ]
         │
         ▼
[ FastAPI Backend (Uvicorn, port 8000) ]
         │  POST /api/predict
         ▼
[ Preprocessing ]
  • Resize to 512 × 512
  • Normalize with ImageNet mean / std
  • Convert to PyTorch tensor (B, C, H, W)
         │
         ▼
[ DeepLabV3 + ResNet-50 Model ]
  • COCO/VOC pre-trained backbone
  • ASPP module (atrous rates: 12, 24, 36)
  • Effective output stride: 8
  • 3-class output logits (1, 3, 512, 512)
         │
         ▼
[ Post-processing ]
  ┌──────────────────────┬──────────────────────────────┐
  │ Segmentation stream  │ Bounding box stream           │
  │                      │                              │
  │ Argmax → class mask  │ cv2.connectedComponentsWithStats │
  │ Nearest-neighbor     │ 8-connectivity               │
  │ upsample to original │ Min-area: 40 px (pothole)    │
  │ resolution           │             25 px (crack)    │
  │ Alpha-blend overlay  │ Confidence: mean softmax prob │
  └──────────────────────┴──────────────────────────────┘
         │
         ▼
[ JSON response: masks, overlay, bboxes, severity, pixel stats ]
         │
         ▼
[ React 18 frontend renders results with interactive view toggle ]
```

---

## Segmentation Classes

| Class ID | Name | Visualization | Description |
|---|---|---|---|
| **0** | Background | Transparent / Black | Undamaged road surface, markings, surroundings. Utility manholes are explicitly mapped here to prevent false alarms. |
| **1** | Pothole | 🔴 Red `#FF3B30` | Discrete asphalt depressions, craters, and cavities |
| **2** | Crack | 🟡 Yellow `#FFCC00` | Longitudinal, transverse, and alligator pavement fissures |

---

## Dataset

The model was trained on a curated, multi-source dataset assembled from six publicly available road damage benchmarks, re-annotated into a unified 3-class polygon format.

### Source Benchmarks

| Dataset | Scenes | 512×512 Crops | Primary Defect Types |
|---|---|---|---|
| CPRID | 45 | 412 | Large potholes, severe surface degradation |
| CNR-Road | 18 | 164 | Complex urban road scenes, shadow variation |
| CRACK500 | 44 | 436 | Longitudinal and transverse fine cracks |
| EdmCrack600 | 45 | 440 | Urban pavement fissures, concrete cracks |
| GAPs384 | 34 | 328 | High-resolution road distress, thin joint cracks |
| Pothole-600 | 44 | 326 | Discrete potholes and edge depressions |
| **Total** | **230 scenes** | **2,106 crops** | |

### Dataset Splits

Splits were performed at the **parent scene level** to prevent spatial leakage — crops from the same physical road location are always in the same split.

| Split | Scenes | Crops |
|---|---|---|
| Training | 90 | 1,304 |
| Validation | 65 | 802 |
| Held-out test | 75 | — |
| **Canonical benchmark** | **17** | **full-frame 640×360** |

The 17-image canonical benchmark consists of independently collected real-world road survey footage, completely separate from all six public benchmark sources. Zero data leakage was verified by MD5 hash comparison across all splits.

### Preprocessing

- Resize to **512 × 512** (training and inference)
- Normalize with ImageNet statistics: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- Synchronised augmentations (training only): random horizontal flip (p=0.5), vertical flip (p=0.2), color jitter, rotation ±10°

---

## Model Training

### Architecture

- **Model**: DeepLabV3 with ResNet-50 backbone
- **Pre-trained weights**: `DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1`
- **Output stride**: 8 (dilated convolutions in ResNet-50 layer3 and layer4)
- **ASPP atrous dilation rates**: 12, 24, 36
- **Auxiliary loss head**: enabled (weight 0.4), tapped off ResNet-50 layer3
- **Output**: 3-channel logits → Softmax → Argmax class assignment

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Backbone learning rate | 1e-5 |
| Classifier head learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 8 |
| Epochs | 30 |
| LR scheduler | Cosine Annealing (min LR 1e-6) |
| Precision | Automatic Mixed Precision (AMP) |
| Loss function | Weighted Cross-Entropy (w=0.4) + Soft Dice (w=0.6) |
| Class weights | Background: 0.2 · Pothole: 1.5 · Crack: 1.5 |

The compound loss function combines **Weighted Cross-Entropy** (to correct class-frequency imbalance, where background constitutes >95% of pixels) with **Soft Dice** (to directly optimize overlap quality and handle class sparsity).

Differential learning rates prevent catastrophic forgetting of pre-trained ResNet-50 feature representations while allowing the ASPP segmentation head to train efficiently from random initialization.

---

## Evaluation Results

Evaluated on the **17-image canonical benchmark** (3,916,800 total pixels, native 640×360 resolution) using the standard argmax decision rule, matching the decoding used during training.

### Quantitative Results

| Class | IoU | Dice (F1) | Precision | Recall |
|---|---|---|---|---|
| **Background** | **96.76%** | **98.35%** | 98.04% | 98.67% |
| **Pothole** | **49.47%** | **66.20%** | 59.74% | 74.22% |
| **Crack** | **35.70%** | **52.62%** | 61.61% | 45.92% |
| **Overall 3-class mIoU** | **60.64%** | **72.39%** | — | — |
| **Damage mIoU** | **42.59%** | **59.41%** | — | — |

### Confusion Matrix (3,916,800 pixels)

```
                  Predicted
              BG       Pothole    Crack
Actual  BG    3,710,676   15,187   34,958
        PH        7,438   23,349      672
        CR       66,795      551   57,174
```

### Severity Assessment (Backend)

The backend classifies road condition based on combined damage percentage:

| Severity | Trigger Condition |
|---|---|
| ✅ Good | Exactly 0.0% damage |
| 🟡 Low | Damage > 0.0% and < 1.5% |
| 🟠 Moderate | Damage ≥ 1.5% and < 5.0% |
| 🔴 High / Critical | Damage ≥ 5.0% |

> **Note on threshold-calibrated mode**: An offline evaluation with per-class probability thresholds ($T_{pothole}=0.30$, $T_{crack}=0.35$) increases pothole recall to 78.76% at a small cost to precision. This mode exists in the evaluation scripts but the production backend uses standard argmax.

---

## Technology Stack

### Backend

| Component | Technology |
|---|---|
| Web framework | FastAPI ≥ 0.110.0 |
| ASGI server | Uvicorn |
| Deep learning | PyTorch ≥ 2.0.0, TorchVision ≥ 0.15.0 |
| Image processing | Pillow ≥ 10.0.0, OpenCV (headless) ≥ 4.8.0 |
| Data validation | Pydantic ≥ 2.0.0 |
| Database | SQLite (via Python `sqlite3`) |
| File upload | python-multipart |

### Frontend

| Component | Technology |
|---|---|
| UI framework | React 18.3 |
| Build tool | Vite 5.1 |
| Styling | Tailwind CSS 3.4 |
| Icons | Lucide React 0.344 |
| Authentication | Firebase (Auth) |

### ML Infrastructure

| Component | Detail |
|---|---|
| Model checkpoint | `best_model.pth` (~160.5 MB) |
| Checkpoint SHA-256 | `2bdac371c33f8b987937b2a3b48d3728f019636081d9a55c1a3cd32b0a5ddedc` |
| Training environment | Kaggle Notebook (NVIDIA GPU) |
| Training script | `build_notebook_v3.py` |

---

## Project Structure

```
ROADVISION AI/
├── backend/
│   ├── main.py                    # FastAPI application, inference pipeline, bounding box engine
│   ├── database.py                # SQLite ORM and inspection schema
│   ├── requirements.txt           # Python dependencies
│   ├── test_backend.py            # Automated backend tests
│   └── samples/                   # Built-in sample road images
│       ├── road_cracks.jpg
│       ├── road_potholes.jpg
│       ├── road_pothole_and_crack.jpg
│       └── road_multiple_damage.jpg
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                # Root component and client-side routing
│       ├── components/
│       │   ├── ImageComparison.jsx  # Multi-mode inspection viewer
│       │   ├── Sidebar.jsx
│       │   ├── Header.jsx
│       │   ├── SeverityBadge.jsx
│       │   └── MetricCard.jsx
│       ├── pages/
│       │   ├── LandingPage.jsx
│       │   ├── LoginPage.jsx
│       │   ├── SignupPage.jsx
│       │   ├── OverviewPage.jsx
│       │   ├── NewInspectionPage.jsx
│       │   ├── DetectionResultsPage.jsx
│       │   ├── InspectionHistoryPage.jsx
│       │   ├── AnalyticsPage.jsx
│       │   └── ReportsPage.jsx
│       ├── context/
│       │   └── AuthContext.jsx
│       └── services/
│           └── firebase.js
│
├── artifacts/
│   └── roadvision_model_artifacts/
│       ├── best_model.pth         # Production model weights (gitignored)
│       ├── model_config.json      # Architecture and normalization config
│       ├── class_mapping.json     # Class IDs, names, and colors
│       └── training_metrics.json  # Recorded evaluation metrics
│
├── notebooks/
│   └── roadvision_kaggle_training.ipynb  # Kaggle training notebook
│
├── build_notebook_v3.py           # Training script used to produce best_model.pth
├── .gitignore
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python ≥ 3.10
- Node.js ≥ 18
- npm or pnpm
- `best_model.pth` placed at `artifacts/roadvision_model_artifacts/best_model.pth`

### 1. Clone the Repository

```bash
git clone https://github.com/PalakChaudhary27/RoadVision-AI.git
cd "RoadVision-AI"
```

### 2. Backend Setup

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

> **Model weights**: `best_model.pth` is excluded from the repository by `.gitignore` due to file size. Place the downloaded checkpoint at `artifacts/roadvision_model_artifacts/best_model.pth` before starting the backend.

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Firebase Configuration

The frontend uses Firebase for authentication. Create `frontend/src/services/firebase.js` (or update the existing file) with your Firebase project credentials. The `.gitignore` excludes `.env` files; store credentials safely.

### 5. Running the Application

**Start the backend** (from the project root, with `.venv` active):

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive documentation is at `http://localhost:8000/docs`.

**Start the frontend** (in a separate terminal, from `frontend/`):

```bash
npm run dev
```

The UI will open at `http://localhost:5173`.

### 6. Quick Test with the Built-in Sample Images

Navigate to **New Inspection** in the sidebar, click any image in the **Sample Gallery**, then click **Run Detection**. No external images needed.

---

## API Reference

All endpoints are served by the FastAPI backend at `http://localhost:8000`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` or `/api/health` | Backend health check; returns model load status and device |
| `POST` | `/predict` or `/api/predict` | Upload a road image and receive segmentation results |
| `GET` | `/api/inspections` | List all saved inspections (supports `?search=` query) |
| `POST` | `/api/inspections` | Save a new inspection record |
| `GET` | `/api/inspections/{id}` | Retrieve a specific inspection by UUID |
| `DELETE` | `/api/inspections/{id}` | Delete a specific inspection |
| `GET` | `/api/analytics` | Aggregated statistics across all saved inspections |
| `GET` | `/api/samples` | List available built-in sample road images |
| `GET` | `/api/samples/{filename}` | Serve a specific sample image file |

### `/api/predict` — Request & Response

**Request**: multipart/form-data with field `file` (JPEG, PNG, or WEBP; max 25 MB).

**Response** (JSON):
```json
{
  "status": "success",
  "image_dimensions": { "width": 640, "height": 360 },
  "pothole_area_percentage": 3.21,
  "crack_area_percentage": 7.84,
  "total_damaged_area_percentage": 11.05,
  "background_area_percentage": 88.95,
  "severity": "High / Critical",
  "severity_description": "Significant pavement degradation requiring prompt repair.",
  "bounding_boxes": [ ... ],
  "mask_base64": "...",
  "overlay_base64": "...",
  "bbox_vis_base64": "..."
}
```

---

## Limitations & Future Work

### Current Limitations

| Limitation | Description |
|---|---|
| **Standing water** | Specular reflections of sky inside potholes alter color histograms, occasionally causing false-negative edge predictions |
| **Harsh shadows** | High-contrast shadow boundaries on worn asphalt can produce false-positive crack detections |
| **No depth measurement** | 2D RGB segmentation cannot quantify physical cavity depth (a shallow surface scratch and a deep structural pothole look similar in pixel area) |
| **CPU inference latency** | At ~41.5 M parameters, DeepLabV3 + ResNet-50 runs at ~450–750 ms per frame on CPU; real-time video requires GPU hardware |
| **Fixed camera geometry** | Damage area is reported as a percentage of image pixels, not in calibrated physical units (m²) |

### Planned Improvements

- **Model compression**: ONNX export and INT8 post-training quantization for deployment on survey vehicle embedded hardware
- **Multi-scale inference**: Test-time augmentation at multiple input resolutions to improve crack recall on thin fissures
- **Stereo / depth integration**: Fuse with monocular depth estimation to report approximate pothole volume in litres
- **Automated reporting**: Generate structured PDF inspection reports with regional damage heat maps
- **Video stream support**: Frame-by-frame analysis with temporal smoothing for dashcam footage

---

## Author

**Palak Chaudhary**  
B.Tech — Computer Science and Engineering  
Lovely Professional University, Phagwara, Punjab, India

[![GitHub](https://img.shields.io/badge/GitHub-PalakChaudhary27-181717?logo=github)](https://github.com/PalakChaudhary27)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-palakchaudhary2717-0A66C2?logo=linkedin)](https://linkedin.com/in/palakchaudhary2717)

---

## License

This project was developed as an academic capstone. See individual dataset licenses for training data usage terms. Model weights and application code are for educational and non-commercial use.
