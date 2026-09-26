# Herring YOLO Fine-Tuning Project — Windows VS Code Version

This project fine-tunes a pretrained YOLO detector on labeled frames from `Herring.mp4`, then predicts/tracks fish in similar videos.

## What this project does

Pipeline:

```text
Herring video
  -> extract frames
  -> manually label fish bounding boxes in YOLO format
  -> split train/val
  -> fine-tune pretrained YOLO last layers using freeze
  -> predict/track on herring video using BoT-SORT
```

This is **not Faster R-CNN**. This is the lightweight route for volunteers:

```text
Pretrained YOLO -> fine-tune final layers -> BoT-SORT tracking -> video prediction
```

## Important requirement: labels are required

The code can extract frames automatically, but students must label fish bounding boxes before training.

Use CVAT, Roboflow, Label Studio, or LabelImg and export labels in YOLO format.

Expected label format, one `.txt` file per image:

```text
class_id x_center y_center width height
```

For this project, use one class:

```text
0 fish
```

Empty frames are allowed. For frames with no visible fish, create an empty `.txt` file with the same image name.

## Folder structure

```text
herring_yolo_finetune_windows/
  videos/
    Herring.mp4
  frames_to_label/
  frames_labeled/
    images/
    labels/
  dataset/
    images/train/
    images/val/
    labels/train/
    labels/val/
  configs/herring.yaml
  scripts/
  runs/
```

## Step 1 — Put video in the folder

Place your video here:

```text
videos/Herring.mp4
```

## Step 2 — Setup environment

Open the folder in VS Code. Open PowerShell terminal and run:

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

This installs:

- Ultralytics YOLO
- OpenCV
- tqdm
- PyTorch dependency through Ultralytics

It also checks whether CUDA GPU is available.

## Step 3 — Extract frames for labeling

```powershell
powershell -ExecutionPolicy Bypass -File run_extract_windows.ps1
```

By default this saves up to 300 frames at about 1 frame per second.

Output:

```text
frames_to_label/
```

## Step 4 — Label the frames

Label fish bounding boxes using CVAT, Roboflow, Label Studio, or LabelImg.

Export YOLO labels and place them like this:

```text
frames_labeled/images/
  herring_000000.jpg
  herring_000030.jpg

frames_labeled/labels/
  herring_000000.txt
  herring_000030.txt
```

If a frame has no fish, create an empty text file with the same stem.

## Step 5 — Split into train/val

```powershell
powershell -ExecutionPolicy Bypass -File run_split_windows.ps1
```

This creates:

```text
dataset/images/train
dataset/images/val
dataset/labels/train
dataset/labels/val
```

## Step 6 — Fine-tune the pretrained YOLO model

```powershell
powershell -ExecutionPolicy Bypass -File run_train_windows.ps1
```

The default model is:

```text
yolov8n.pt
```

This downloads automatically the first time.

The script uses:

```text
--freeze 10
```

That freezes the earlier layers and trains mainly the later layers/head.

Output weights:

```text
runs/train/herring_yolo_finetune/weights/best.pt
```

## Step 7 — Predict and track on video

```powershell
powershell -ExecutionPolicy Bypass -File run_predict_windows.ps1
```

This runs:

```text
best.pt + botsort.yaml
```

Output:

```text
runs/predict/herring_video_prediction/
```

## To use YOLO11 instead

Edit `run_train_windows.ps1` and replace:

```text
yolov8n.pt
```

with:

```text
yolo11n.pt
```

## To use the old project model

If `current_fish_model.pt` is Ultralytics-compatible, place it in the project root and edit `run_train_windows.ps1`:

```powershell
python scripts\train_yolo_finetune.py --model current_fish_model.pt --data configs\herring.yaml --epochs 50 --imgsz 640 --batch 8 --freeze 10 --device auto --workers 0
```

## GPU notes

- On Windows with NVIDIA GPU, the script uses CUDA if PyTorch detects it.
- If CUDA is not available, it falls back to CPU, which will be slower.
- Reduce `--batch` to 4 or 2 if GPU memory is low.
- Reduce `--imgsz` to 416 if memory is still low.

## Recommended first training settings

For volunteer laptops:

```text
epochs: 30-50
batch: 4-8
imgsz: 640 or 416
freeze: 10
model: yolov8n.pt
```

For stronger GPU:

```text
epochs: 100
batch: 16
imgsz: 640
freeze: 10 first, then freeze: 0 for full fine-tuning
```
