.\.venv\Scripts\Activate.ps1
python scripts\split_yolo_dataset.py --source-images frames_labeled\images --source-labels frames_labeled\labels --dataset-dir dataset --val-ratio 0.2
