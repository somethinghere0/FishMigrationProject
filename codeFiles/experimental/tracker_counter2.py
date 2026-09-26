import cv2
from ultralytics import solutions

cap = cv2.VideoCapture("testVideos/shortClip.mp4")
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

# Init trackzone
trackzone = solutions.TrackZone(
    show=True,
    region=region_points,
    model="current_fish_model.pt",
)

# Exit tracking variables
exit_counter = 0
tracked_objects = {}  # Store previous positions: {track_id: x_position}
exited_ids = set()  # Track IDs that have already exited

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = trackzone(im0)
    
    # Access the underlying YOLO results from the tracker
    if hasattr(trackzone, 'track_history') and trackzone.track_history:
        # Get current frame detections with tracking IDs
        for track_id, positions in trackzone.track_history.items():
            if len(positions) > 0:
                current_x, current_y = positions[-1]  # Most recent position
                
                # Check if object crosses left boundary
                if track_id in tracked_objects and track_id not in exited_ids:
                    prev_x = tracked_objects[track_id]
                    
                    # If object was inside zone and is now past left boundary
                    if prev_x >= left and current_x < left:
                        exit_counter += 1
                        exited_ids.add(track_id)
                        print(f"Object {track_id} exited! Total exits: {exit_counter}")
                
                # Update tracked position
                tracked_objects[track_id] = current_x
    
    # Draw exit counter on frame
    frame_with_counter = results.plot_im.copy()
    cv2.putText(frame_with_counter, f"Exits: {exit_counter}", 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    video_writer.write(frame_with_counter)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"\nFinal exit count: {exit_counter}")