# CFD RF-DETR Fine-Tuning on Herring Video Frames

This VS Code project fine-tunes a **Community Fish Detector RF-DETR checkpoint** on your own labeled frames from `Herring.mp4`, then predicts fish boxes on similar herring videos.

The goal is:

```text
Herring.mp4
  -> extract representative frames
  -> label fish bounding boxes
  -> prepare YOLO-format dataset
  -> fine-tune CFD RF-DETR checkpoint
  -> predict on video/images
```

## Important clarification

This project fine-tunes the **RF-DETR version of Community Fish Detector**. It is not an Ultralytics YOLO + BoT-SORT project. The CFD repo currently recommends RF-DETR models; the older YOLOv12x route is deprecated in that repo.

RF-DETR's high-level training API does not expose a simple `freeze last layers only` switch. To make this behave like conservative last-layer adaptation, this project:

- starts from a pretrained CFD checkpoint,
- uses a low learning rate,
- uses an even lower encoder/backbone learning rate when supported,
- uses Nano first for lower memory.

That is the safest student-friendly equivalent of “unlock only the last few layers” without rewriting the RF-DETR Lightning training loop.

## Folder structure

```text
videos/Herring.mp4                         # students put video here
weights/                                   # students put CFD checkpoint here
frames_raw/Herring/                         # extracted frames
frames_labeled/images/                      # labeled images copied here
frames_labeled/labels/                      # YOLO labels copied here
datasets/herring_yolo/                      # train/valid/test dataset generated here
outputs/cfd_herring_finetune/               # fine-tuned checkpoints
outputs/Herring_finetuned_prediction.mp4    # prediction output
```

## Step 1 — Put the video in place

Place the video here:

```text
videos/Herring.mp4
```

## Step 2 — Download CFD weights

Download one RF-DETR checkpoint from the Community Fish Detector releases page:

<https://github.com/filippovarini/community-fish-detector/releases>

Start with Nano for student machines:

```text
cfd-2026.02.02-rf-detr-nano-640.pth
```

Put it here:

```text
weights/cfd-2026.02.02-rf-detr-nano-640.pth
```

If you use Small or Medium, update `configs/train_config.yaml`.

## Step 3 — Setup environment

### Windows

Open VS Code terminal inside this folder:

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

For NVIDIA GPU, install the correct PyTorch CUDA wheel first from:

<https://pytorch.org/get-started/locally/>

Example for CUDA 12.1:

```powershell
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Mac

```bash
bash setup_mac.sh
```

Mac CPU training will be slow. Apple Silicon MPS may help depending on PyTorch/RF-DETR support, but for serious training use Colab/lab NVIDIA GPU.

## Step 4 — Extract frames

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File run_1_extract_windows.ps1
```

### Mac

```bash
bash run_1_extract_mac.sh
```

Default: save 300 frames, one every ~30 frames. For a more diverse dataset, extract from multiple videos and include empty/hard frames.

## Step 5 — Label fish boxes

Use Roboflow, CVAT, Label Studio, or LabelImg.

Label class:

```text
0 fish
```

Export YOLO format:

```text
class_id x_center y_center width height
```

Copy labeled images to:

```text
frames_labeled/images/
```

Copy YOLO `.txt` labels to:

```text
frames_labeled/labels/
```

For images with no fish, create an empty `.txt` file with the same base name. Empty frames are important because they reduce false positives from bubbles, glare, debris, and turbulence.

## Step 6 — Prepare train/valid/test dataset

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File run_2_prepare_windows.ps1
```

### Mac

```bash
bash run_2_prepare_mac.sh
```

This creates:

```text
datasets/herring_yolo/
  data.yaml
  train/images, train/labels
  valid/images, valid/labels
  test/images, test/labels
```

## Step 7 — Fine-tune CFD RF-DETR

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File run_3_train_windows.ps1
```

### Mac

```bash
bash run_3_train_mac.sh
```

The default config is conservative:

```yaml
model_size: nano
epochs: 25
batch_size: 2
grad_accum_steps: 8
lr: 0.00005
lr_encoder: 0.00001
```

If you get out-of-memory:

```yaml
batch_size: 1
grad_accum_steps: 16
```

The best checkpoint should be saved as something like:

```text
outputs/cfd_herring_finetune/checkpoint_best_total.pth
```

## Step 8 — Predict on a similar video

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File run_4_predict_windows.ps1
```

### Mac

```bash
bash run_4_predict_mac.sh
```

Outputs:

```text
outputs/Herring_finetuned_prediction.mp4
outputs/Herring_finetuned_detections.csv
```

This script predicts bounding boxes and visible fish detections per frame. It does not do BoT-SORT tracking. Use it first to verify that fine-tuning improves detection. Tracking/counting can be added after the detector is stable.

## Student checklist

1. Confirm the video opens.
2. Extract representative frames.
3. Label fish boxes.
4. Include empty frames.
5. Prepare the dataset.
6. Confirm `data.yaml` exists.
7. Download CFD RF-DETR weights.
8. Train with Nano first.
9. Predict on the original or a new herring video.
10. Compare before/after qualitatively and with detection CSV counts.

## Suggested small first dataset

Use about 150–300 labeled images:

- 70% clear fish frames,
- 15% empty frames,
- 10% difficult frames with glare/debris/turbidity,
- 5% dense/overlapping fish frames.

For a real generalization experiment, do not train and test on near-duplicate frames from the same video. Add multiple videos and split by video.


## Single-class label/class-ID fix

This version includes three fixes for the herring project:

1. `scripts/prepare_yolo_dataset.py` now writes `nc: 1` into `data.yaml`.
2. `scripts/validate_yolo_labels.py` checks that every YOLO label uses only class `0`.
3. `scripts/predict_video_rfdetr.py` and `scripts/predict_images_rfdetr.py` record any raw RF-DETR class ID in the CSV, but force the displayed/used class to `0 = fish` because this is a single-class counting project.

Run after preparing the dataset:

Windows:
```powershell
powershell -ExecutionPolicy Bypass -File run_validate_windows.ps1
```

Mac:
```bash
bash run_validate_mac.sh
```
