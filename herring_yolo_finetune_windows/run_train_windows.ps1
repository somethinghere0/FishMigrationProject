.\.venv\Scripts\Activate.ps1
# Uses GPU automatically if CUDA is available. If not, it falls back to CPU.
# For a custom previous model, replace yolov8n.pt with current_fish_model.pt.
python scripts\train_yolo_finetune.py --model yolo11n.pt --data configs\herring.yaml --epochs 50 --imgsz 640 --batch 8 --freeze 10 --device auto --workers 0
