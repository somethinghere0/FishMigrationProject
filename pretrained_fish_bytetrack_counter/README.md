# Pretrained Fish Detector → ByteTrack/BoT-SORT-Style Tracking → Herring Count

This VS Code project is designed for volunteers who **do not have enough GPU memory to train Faster R-CNN from scratch**.

Instead, this pipeline runs:

```text
Pretrained fish detector
        ↓
fish bounding boxes per frame
        ↓
ByteTrack-style tracking
        ↓
line-crossing herring count
        ↓
CSV + annotated video outputs
```

The recommended first detector is **Community Fish Detector (CFD)** using RF-DETR pretrained weights. You can also use an Ultralytics YOLO `.pt` model as a fallback or comparison.

---

## Why this is different from Faster R-CNN training

The previous code trained a Faster R-CNN detector from scratch. That is useful for teaching, but it is memory-heavy.

This code does **no detector training**. It uses pretrained fish detector weights and only performs inference, tracking, and counting. This makes it much easier for students to run on laptops.

---

## Project structure

```text
pretrained_fish_bytetrack_counter/
    configs/
        cfd_counter.yaml
        ultralytics_counter.yaml
    data/
        videos/
            put your .mp4/.mov/.avi files here
    outputs/
        results are written here
    src/fish_counter/
        detectors.py
        pipeline.py
        preprocess.py
        tracker.py
        utils.py
    run_counter.py
    evaluate_counts.py
    requirements.txt
    manual_counts_template.csv
```

---

## Setup in VS Code

Open this folder in VS Code. Then in the VS Code terminal:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Mac/Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `rfdetr` fails on a volunteer machine, use the Ultralytics fallback config first:

```bash
python run_counter.py --config configs/ultralytics_counter.yaml
```

That fallback will run YOLO-style inference and the same custom tracking/counting logic.

---

## Download Community Fish Detector weights

Weights are not included in this ZIP.

Download RF-DETR CFD weights from the Community Fish Detector GitHub releases page:

```text
https://github.com/filippovarini/community-fish-detector/releases
```

Recommended first model for laptops:

```text
RF-DETR Nano, 640 input size
```

Put the downloaded `.pth` file in:

```text
weights/
```

Then edit `configs/cfd_counter.yaml`:

```yaml
model:
  backend: cfd_rfdetr
  weights: weights/YOUR_DOWNLOADED_MODEL.pth
  resolution: 640
  confidence: 0.30
```

For CFD Small or Medium models, use:

```yaml
resolution: 1024
```

---

## Put videos into the project

Place herring-counting videos in:

```text
data/videos/
```

Example:

```text
data/videos/maine_herring_001.mp4
```

---

## Run CFD + tracker + count

```bash
python run_counter.py --config configs/cfd_counter.yaml
```

Outputs will be saved to:

```text
outputs/cfd_results/
    annotated_videos/
    detections_tracks.csv
    video_counts.csv
```

`video_counts.csv` contains one predicted count per video.

---

## Run YOLO fallback / comparison

This is useful if RF-DETR installation is difficult or if you have a fish-pretrained YOLO `.pt` file.

Edit `configs/ultralytics_counter.yaml`:

```yaml
model:
  backend: ultralytics
  weights: yolo11n.pt
  confidence: 0.25
```

Then run:

```bash
python run_counter.py --config configs/ultralytics_counter.yaml
```

If you have a fish-specific YOLO model, replace `yolo11n.pt` with that path.

---

## Set the counting line correctly

The default counting line is:

```yaml
counting:
  line: [320, 0, 320, 720]
  direction: right
```

This means:

```text
x1=320, y1=0, x2=320, y2=720
```

So the line is vertical. A fish is counted when its track crosses the line from left to right.

Change direction depending on your video:

```yaml
direction: right
```

or:

```yaml
direction: left
```

or:

```yaml
direction: up
```

or:

```yaml
direction: down
```

or:

```yaml
direction: any
```

For herring-counter videos, manually inspect the annotated output and adjust this line until it matches the human counting rule.

---

## Preprocessing options

The pipeline includes the preprocessing techniques your team already discussed:

```yaml
preprocess:
  enabled: true
  gray_world_white_balance: true
  clahe: true
  gamma: 1.05
  denoise: false
  sharpen: false
```

Recommended first run:

```yaml
gray_world_white_balance: true
clahe: true
gamma: 1.05
```

If the videos are noisy, try:

```yaml
denoise: true
```

If the fish edges are weak, try:

```yaml
sharpen: true
```

Do not turn on every enhancement blindly. Compare the annotated videos and count errors.

---

## Evaluate against manual counts

Create a CSV like this:

```csv
source,manual_count
data/videos/maine_herring_001.mp4,24
data/videos/maine_herring_002.mp4,0
```

Then run:

```bash
python evaluate_counts.py \
  --pred outputs/cfd_results/video_counts.csv \
  --manual manual_counts.csv \
  --out outputs/cfd_results/count_error_report.csv
```

This gives:

- predicted count
- manual count
- signed error
- absolute error
- percent error
- MAE
- MAPE

---

## What volunteers should report

For each video, ask the students to report:

```text
1. video name
2. manual count
3. CFD predicted count
4. absolute error
5. whether fish were missed, double-counted, or false detections
6. screenshot of annotated output
7. preprocessing settings used
8. confidence threshold used
9. counting line used
```

---

## Important scientific note

This is not a herring-specific trained model yet. This is a **fish-pretrained generalization test**.

The research question is:

```text
Can a general pretrained fish detector detect and count herring in fish-ladder/counter videos without additional training?
```

If the first result is weak, the next step is not failure. The next step is:

```text
Fine-tune the detector on MIT/Maine herring frames and rerun the same counting pipeline.
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named cv2`

Run:

```bash
pip install opencv-python
```

### `ModuleNotFoundError: No module named tqdm`

Run:

```bash
pip install tqdm
```

### CFD weights not found

Check the path in:

```text
configs/cfd_counter.yaml
```

The model file must exist locally.

### RF-DETR install is too heavy

Use:

```bash
python run_counter.py --config configs/ultralytics_counter.yaml
```

This lets the students test the entire counting pipeline before solving RF-DETR installation.

---

## Recommended first experiments

| Experiment | Detector | Training? | Goal |
|---|---|---:|---|
| 1 | CFD RF-DETR Nano | No | First fish-pretrained result |
| 2 | CFD RF-DETR Small/Medium | No | Higher accuracy if hardware allows |
| 3 | YOLO11n or YOLOv8n | No | Baseline/fallback |
| 4 | Fish-specific YOLO `.pt` | No | Compare with YOLO pretrained fish detector |
| 5 | Fine-tuned herring model | Yes | Improve generalization |

