.\.venv\Scripts\Activate.ps1
python scripts\predict_video_yolo.py --weights C:\Users\zavan\Downloads\FishProject\herring_yolo_finetune_windows\runs\detect\runs\train\herring_yolo_finetune-6\weights\best.pt --source videos\Herring.mp4 --tracker botsort.yaml --conf 0.25 --device auto
