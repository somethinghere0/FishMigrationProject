import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("correction.mp4")
assert cap.isOpened(), "Error reading video file"

# Get output filename from user
output_filename = input("Enter output video filename (without extension): ").strip()
if not output_filename:
    output_filename = "object_counting_output"
# Ensure .mp4 extension
if not output_filename.endswith(('.mp4', '.avi')):
    output_filename += ".mp4"

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points as a centered rectangle in the frame
center_x, center_y = w // 2, h // 2
rect_width = int(w * 0.6)   # 60% of frame width
rect_height = int(h * 0.6)  # 60% of frame height

left = center_x - rect_width // 2
right = center_x + rect_width // 2
top = center_y - rect_height // 2
bottom = center_y + rect_height // 2

region_points = [(left, top), (right, top), (right, bottom), (left, bottom)]

# Video writer
video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init trackzone (object tracking in zones, not complete frame)
trackzone = solutions.TrackZone(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="current_fish_model.pt",  # use any model that Ultralytics support, i.e. YOLOv9, YOLOv10
    # line_width=2,  # adjust the line width for bounding boxes and text display
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = trackzone(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the video file

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows