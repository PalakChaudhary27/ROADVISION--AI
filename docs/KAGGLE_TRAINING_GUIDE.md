# RoadVision AI - Kaggle Cloud GPU Training Guide

This guide provides step-by-step instructions for running the **RoadVision AI** road damage segmentation model training pipeline on **Kaggle's free Cloud GPU** (NVIDIA Tesla T4 or P100).

---

## 1. Prerequisites

1. A free account on [Kaggle](https://www.kaggle.com).
2. Phone verification completed on Kaggle (required by Kaggle to enable free GPU accelerators: **30 hours/week**).
3. The training notebook file from this repository:
   `notebooks/roadvision_kaggle_training.ipynb`

---

## 2. Recommended Verified Datasets

The notebook is configured to automatically detect and ingest road damage segmentation datasets containing both **potholes** and **cracks**.

### Primary Recommended Dataset (Direct Kaggle 1-Click Access)
* **Dataset Name:** `Road damage dataset for segmentation`
* **Creator / Kaggle Slug:** `khoithanhnguyen/road-damage-dataset-for-segmentation`
* **URL:** [https://www.kaggle.com/datasets/khoithanhnguyen/road-damage-dataset-for-segmentation](https://www.kaggle.com/datasets/khoithanhnguyen/road-damage-dataset-for-segmentation)
* **Contents:** 5,770 road damage images annotated with pixel-level polygon segmentation masks for both **potholes** and **cracks**.
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

### Verified Alternatives
1. **Cracks and Potholes in Road Images (CPRID)** by Passos et al. (UNIVALI / DNIT)
   * Original DOI: [10.17632/t576ydh9v8.4](https://doi.org/10.17632/t576ydh9v8.4)
   * 2,235 highway pavement images with discrete binary PNG masks for potholes, cracks, and road lane.
2. **Road Damage Dataset: Potholes, Cracks and Manholes**
   * Kaggle Slug: `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`
   * Over 2,000 real-world smartphone and GoPro images of road defects.

---

## 3. Step-by-Step Training Instructions

### Step 1: Upload the Notebook to Kaggle
1. Go to [kaggle.com](https://www.kaggle.com) and log in.
2. In the left navigation menu, click **Create** (+) $\rightarrow$ **New Notebook**.
3. In the top notebook menu, click **File** $\rightarrow$ **Import Notebook**.
4. In the dialog, select the **Upload** tab.
5. Drag and drop or browse to `notebooks/roadvision_kaggle_training.ipynb` from your local `ROADVISION AI` repository.
6. Click **Import**.

---

### Step 2: Enable the GPU Accelerator
1. In the right-hand panel of your Kaggle notebook, locate the **Notebook settings** section.
2. Find the **Accelerator** dropdown.
3. Change the selection from **None** to **GPU T4 x2** (or **GPU P100**).
4. Verify that the session restarts and displays the GPU accelerator status at the top-right (e.g., `GPU: 0% / 16 GB`).

> **Tip:** Make sure **Internet** is toggled **ON** under the right panel settings so that standard packages and weights can resolve without restriction.

---

### Step 3: Attach the Dataset
1. In the right-hand panel, click the **+ Add Input** button (or **+ Add Data**).
2. In the search bar of the popup modal, type:
   ```text
   road damage dataset for segmentation
   ```
   or search by slug:
   ```text
   khoithanhnguyen/road-damage-dataset-for-segmentation
   ```
3. Click the **+** (Add) button next to the dataset.
4. Once attached, verify that `/kaggle/input/road-damage-dataset-for-segmentation/` appears in the **Input** explorer on the right panel.

---

### Step 4: Run the Training Pipeline
1. In the top menu, click **Run All** (or press `Ctrl + F10` / `Shift + Enter` cell by cell).
2. The notebook will automatically:
   - Verify CUDA GPU availability and memory.
   - Scan `/kaggle/input` and discover all image-mask pairs.
   - Display a preview of road images and ground truth masks.
   - Deterministically split data into Train (70%), Validation (15%), and Test (15%) subsets.
   - Initialize the **RoadVision U-Net** semantic segmentation model.
   - Train using Automatic Mixed Precision (`torch.cuda.amp`) with Cosine Annealing learning rate.
   - Save the best model checkpoint based on validation Mean IoU (`best_model.pth`).
   - Plot loss curves, Mean IoU, and class-wise IoU trajectories.
   - Perform test set evaluation and display qualitative diagnostic overlays.
   - Export all artifacts to `/kaggle/working/roadvision_artifacts/` and bundle them into `roadvision_model_artifacts.zip`.

---

### Step 5: Download Trained Model Artifacts
After training finishes:
1. In the right-hand panel, look at the **Output** section (`/kaggle/working/`).
2. You will see:
   - `roadvision_model_artifacts.zip`
   - `roadvision_artifacts/`
     - `best_model.pth`
     - `model_config.json`
     - `class_mapping.json`
     - `training_metrics.json`
     - `training_curves.png`
     - `test_predictions_sample.png`
3. Hover over `roadvision_model_artifacts.zip` (or the `roadvision_artifacts` folder), click the **three dots (...)**, and select **Download**.
4. Extract the downloaded archive into your local `ROADVISION AI/models/` or `ROADVISION AI/configs/` directory for FastAPI inference.

---

## 4. Hyperparameter Customization

If you want to customize the training session, modify the `CONFIG` block in **Cell 5** of the notebook:

| Parameter | Default | Recommended Adjustment |
| :--- | :--- | :--- |
| `batch_size` | `8` | Use `16` on P100 GPU; reduce to `4` if handling images $>512\times 512$. |
| `num_epochs` | `35` | Increase to `50` for thorough convergence, or `15-20` for a rapid test run. |
| `image_size` | `(512, 512)` | Keep `(512, 512)` for road cracks; reduce to `(384, 384)` for ultra-fast training. |
| `learning_rate` | `1e-4` | Standard for AdamW with U-Net. |
| `dice_loss_weight` | `0.6` | Balances region overlap vs pixel cross-entropy (`0.4`). |
| `class_weights` | `[0.2, 1.5, 1.5]` | Emphasizes rare pothole and crack classes over dominant background. |

---

## 5. Resuming an Interrupted Session

Kaggle notebooks have a 12-hour continuous execution limit. If your training is paused or interrupted:
1. Checkpoint files are saved continuously to `/kaggle/working/checkpoints/checkpoint_latest.pth`.
2. Simply re-run the notebook; **Cell 23** will automatically detect `checkpoint_latest.pth` and resume training from the exact saved epoch and optimizer state.
