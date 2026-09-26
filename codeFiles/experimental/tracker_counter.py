import cv2
import numpy as np

from ultralytics import solutions

cap = cv2.VideoCapture("correction.mp4")
assert cap.isOpened(), "Error reading video file"

# Get output filename from user
output_filename = input("Enter output video filename (without extension): ").strip()
if not output_filename:
    output_filename = "tracker_counter_output"
# Ensure .mp4 extension
if not output_filename.endswith(('.mp4', '.avi')):
    output_filename += ".mp4"

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points for TrackZone (centered rectangle)
center_x, center_y = w // 2, h // 2
rect_width = int(w * 0.6)   # 60% of frame width
rect_height = int(h * 0.6)  # 60% of frame height

left = center_x - rect_width // 2
right = center_x + rect_width // 2
top = center_y - rect_height // 2
bottom = center_y + rect_height // 2

trackzone_region = [(left, top), (right, top), (right, bottom), (left, bottom)]

# Define region points for ObjectCounter (vertical line in center)
box_width = 10  # width of the vertical box in pixels
box_top = 0
box_bottom = int(h)
counter_center_x = w // 2
counter_left_x = counter_center_x - box_width // 2 - 100
counter_right_x = counter_center_x + box_width // 2 - 100

# Vertical line region for counting (two points define a line)
counter_region = [
    (counter_left_x, box_top),      # top-left
    (counter_left_x, box_bottom)    # bottom-left
]

# Video writer
video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize TrackZone (object tracking in zones)
trackzone = solutions.TrackZone(
    show=True,  # display the output
    region=trackzone_region,  # pass region points
    model="current_fish_model.pt",  # use any model that Ultralytics support, i.e. YOLOv9, YOLOv10
    # line_width=2,  # adjust the line width for bounding boxes and text display
)

# Initialize ObjectCounter (object counting across a line/region)
counter = solutions.ObjectCounter(
    show=True,  # display the output
    region=counter_region,  # pass region points
    model='current_fish_model.pt',  # model="yolo11n-obb.pt" for object counting with OBB model.
    # classes=[0, 2],  # count specific classes, i.e., person and car with the COCO pretrained model.
    tracker="botsort.yaml",  # choose trackers i.e "bytetrack.yaml"
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    # Process frame through TrackZone (tracks objects within the zone)
    # This maintains trackzone's internal tracking state
    trackzone_results = trackzone(im0.copy())
    
    # Process the same original frame through ObjectCounter (counts objects crossing the line)
    # This maintains counter's internal counting state
    counter_results = counter(im0.copy())
    
    # Combine both visualizations
    # Use trackzone's output as base (shows tracking within the zone with bounding boxes and IDs)
    combined_frame = trackzone_results.plot_im.copy()
    
    # Overlay the counter's counting line (blue vertical line)
    cv2.line(combined_frame, 
             counter_region[0], 
             counter_region[1], 
             color=(255, 0, 0),  # Blue color for counter line
             thickness=3)
    
    # Overlay counter's count text annotations
    # Copy the counter frame and extract text regions (typically in corners)
    counter_frame = counter_results.plot_im.copy()
    
    # Extract counter's annotations by copying regions that differ significantly
    # This preserves the count numbers and labels from ObjectCounter
    diff = cv2.absdiff(combined_frame, counter_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, text_mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
    
    # Dilate mask slightly to capture text better
    kernel = np.ones((3, 3), np.uint8)
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)
    
    # Overlay counter's text annotations onto combined frame
    text_mask_3channel = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR)
    combined_frame = np.where(text_mask_3channel > 128, counter_frame, combined_frame)
    
    # Ensure both region lines are clearly visible
    # Redraw trackzone region (green rectangle)
    trackzone_pts = np.array(trackzone_region, np.int32)
    trackzone_pts = trackzone_pts.reshape((-1, 1, 2))
    cv2.polylines(combined_frame, [trackzone_pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Redraw counter line (blue vertical line)
    cv2.line(combined_frame, 
             counter_region[0], 
             counter_region[1], 
             color=(255, 0, 0), 
             thickness=3)
    
    # Write the combined frame
    video_writer.write(combined_frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows

