# MIT Sea Grant River Herring Detection + BoT-SORT Counting Project

This is a VS Code-ready PyTorch teaching project for detecting, tracking, and counting river herring/fish from the MIT Sea Grant River Herring dataset.

## What students should learn

1. **Detection is trained; BoT-SORT is mostly not trained.**  
   BoT-SORT is a tracking-by-detection method. You train a detector to find fish in each frame. The tracker then links detections across frames using motion, IoU matching, and optionally appearance/ReID features.

2. **Counting has two cases.**
   - Image counting: count detected fish boxes after score thresholding and NMS.
   - Video counting: detect fish per frame, track IDs across frames, then count unique confirmed tracks or line crossings.

3. **Training from scratch is educational but difficult.**  
   The default model uses no COCO pretrained weights. This lets students understand true scratch training. For research-grade performance, set `pretrained_backbone: true` in `configs/default.yaml` and fine-tune.

4. **Preprocessing matters underwater.**  
   The project includes gray-world white balance, CLAHE contrast enhancement, gamma correction, optional denoising, optional sharpening, and optional barrel/fisheye distortion correction from calibration parameters.

---

## Dataset notes

The public LILA release of the MIT Sea Grant River Herring dataset uses **COCO annotations**. Put the dataset here:

```text
data/raw/mit-river-herring/
  metadata.json
  Coonamessett/...
  Ipswich/...
  Santuit/...
```

The dataset is large. For a classroom demo, start with a subset rather than the full 40 GB image archive.

---

## Setup in VS Code

```bash
cd mit_herring_botsort_project
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Open the folder in VS Code and select the `venv` Python interpreter.

---

## Recommended workflow

### 1. Sanity-check a small subset

```bash
python prepare_dataset.py --config configs/default.yaml --limit 200
```

For training, keep `paths.raw_root` pointing to the original images unless you intentionally update `paths.coco_json` to the preprocessed JSON.

### 2. Train detector from scratch

```bash
python train.py --config configs/default.yaml
```

Outputs:

```text
runs/mit_herring/best.pt
runs/mit_herring/last.pt
```

The validation log reports:

- precision
- recall
- F1
- count MAE, meaning average absolute fish-count error per image

### 3. Count fish in uploaded images

```bash
python infer_images.py \
  --config configs/default.yaml \
  --checkpoint runs/mit_herring/best.pt \
  --input path/to/uploaded_images \
  --output runs/uploaded_image_counts
```

Outputs:

```text
runs/uploaded_image_counts/image_counts.csv
runs/uploaded_image_counts/*_counted.jpg
```

### 4. Track and count fish in video

```bash
python track_count_video.py \
  --config configs/default.yaml \
  --checkpoint runs/mit_herring/best.pt \
  --video path/to/video.mp4 \
  --output runs/video_count
```

Outputs:

```text
runs/video_count/tracked_counted.mp4
runs/video_count/video_counts_by_frame.csv
```

### 5. Hyperparameter tuning

```bash
python hparam_tune.py --config configs/default.yaml --trials 10 --epochs-per-trial 3
```

Start small. After students understand tuning, increase trials, epochs, and subset size.

---

## Important hyperparameters

| Area | Hyperparameter | What it controls |
|---|---|---|
| Optimizer | `lr` | Step size; too high diverges, too low learns slowly |
| Optimizer | `weight_decay` | Regularization against overfitting |
| Data | `batch_size` | Memory use and gradient stability |
| Detection | `score_threshold` | Fish count sensitivity; lower detects more but adds false positives |
| Detection | `nms_threshold` | Removes duplicate boxes around the same fish |
| Tracking | `iou_match_threshold` | How similar boxes must be to keep the same track ID |
| Tracking | `max_age` | How long a fish track can disappear before being removed |
| Tracking | `min_hits` | How many frames before a track is trusted |

---

## Preprocessing choices for MIT herring footage

Use these as controlled experiments, not all at once blindly:

1. **Gray-world white balance**: fixes blue/green underwater color cast.
2. **CLAHE on LAB luminance**: improves local contrast in low-visibility water.
3. **Gamma correction**: helps dim or nighttime frames.
4. **Denoising**: useful for turbid/noisy footage, but can blur small fish.
5. **Unsharp masking**: can reveal fish edges, but may exaggerate water particles.
6. **Barrel/fisheye correction**: only use if you have camera calibration parameters. Guessing distortion coefficients can make annotations misalign.

For rigorous experiments, train/evaluate these variants:

- raw frames only
- raw + white balance
- raw + white balance + CLAHE
- raw + white balance + CLAHE + gamma
- full preprocessing, only if calibration is known

Report both detection F1 and count MAE. The best detector is not always the best counter.

---

## How to explain BoT-SORT to students

BoT-SORT is not a single model that learns fish from images. It is a pipeline:

```text
Video frame -> fish detector -> bounding boxes -> tracker association -> fish IDs -> count
```

The detector must learn fish appearance. The tracker solves: “Is this fish in frame t the same fish as a fish in frame t-1?”

The included tracker is a BoT-SORT-inspired educational implementation using:

- Kalman filter motion prediction
- IoU association
- Hungarian matching
- track confirmation using `min_hits`
- stale-track deletion using `max_age`

For a research-grade BoT-SORT, add:

- camera motion compensation
- appearance/ReID embedding model
- confidence-aware association

---

## Expected student experiments

1. Train scratch detector vs pretrained-backbone detector.
2. Compare preprocessing settings.
3. Sweep `score_threshold` from 0.2 to 0.7 and plot count MAE.
4. Tune `iou_match_threshold`, `max_age`, and `min_hits` on validation videos.
5. Report failure cases:
   - overlapping fish
   - blurry fish
   - nighttime reflections
   - non-herring species
   - occlusions near rocks, leaves, bubbles, or water artifacts

---

## Practical recommendation

For a first class run, use scratch training for learning and a pretrained backbone for the best final count. Scratch training may require much more data, many epochs, and strong augmentation to compete with fine-tuning.
