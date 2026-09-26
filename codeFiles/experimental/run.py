# from ultralytics import YOLO

# # # Load a model
# # model = YOLO("yolo11n.yaml")  # build a new model from YAML
# # model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = 'yolo11n_fish_trained.pt'
# results = model('image.jpg',save =True)
# for r in results:
#   print(f"Detected {len(r)} objects in image")
# Train the model
# results = model.train(data="./roboflow/data.yaml", epochs=100, imgsz=640)


from ultralytics import YOLO
import os

# 1. This is just a STRING path
model_path = 'cfd-yolov12x-1.00.pt'

# Check if the file actually exists
if os.path.exists(model_path):
    
    # 2. --- THIS IS THE MISSING STEP ---
    # Load the model path string into the YOLO class
    # to create a real, callable model OBJECT.
    model = YOLO(model_path) 

    # 3. Now you can "call" the model OBJECT to run predictions

    #training
    # resultst = model.train(data="Deepfish_data.yaml", epochs=100, imgsz=640)

    #track
    results = model.track('testVideos/aquarium2bw.mp4', save=True, conf=0.5, tracker="botsort.yaml")

    # just detection
    # results = model('testVideos/bw_slow_cropped1.mp4', save=True, conf=0.5)
    
    for r in results:
      print(f"Detected {len(r.boxes)} objects in image") # A small correction to be more precise

else:
    print(f"Error: Model file not found at '{model_path}'")
    print("Please make sure 'yolo11n_fish_trained.pt' is in the correct directory.")