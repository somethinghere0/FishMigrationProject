# Herring Frame Extractor - Windows VS Code Version

This project extracts frames from `Herring.mp4` using Python and OpenCV.

## 1. Open in VS Code

Open this folder in VS Code:

```text
herring_frame_extractor_windows
```

## 2. Add the video

Place your video here:

```text
videos/Herring.mp4
```

If your video is named `HerringVideo.mp4`, either rename it to `Herring.mp4` or update the command in `run_windows.ps1`.

## 3. Open PowerShell terminal in VS Code

In VS Code:

```text
Terminal → New Terminal
```

## 4. Run setup

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

## 5. Extract frames

```powershell
powershell -ExecutionPolicy Bypass -File run_windows.ps1
```

Frames will be saved to:

```text
extracted_frames/Herring/
```

## Useful commands

Save about 1 frame per second for a 30 FPS video:

```powershell
python extract_frames.py --video videos/Herring.mp4 --output extracted_frames/Herring --interval 30
```

Save every 10th frame:

```powershell
python extract_frames.py --video videos/Herring.mp4 --output extracted_frames/Herring_interval10 --interval 10
```

Save every frame:

```powershell
python extract_frames.py --video videos/Herring.mp4 --output extracted_frames/Herring_all --interval 1
```

Resize extracted frames to width 640:

```powershell
python extract_frames.py --video videos/Herring.mp4 --output extracted_frames/Herring_640 --interval 30 --resize-width 640
```

## Recommendation

For the first dataset, use:

```text
--interval 30
```

This gives roughly 1 frame per second for a 30 FPS video, which is manageable for volunteers.
