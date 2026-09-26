import cv2
import json
import numpy as np
# You'll use the get_mapping function from our previous cohesive script here

def save_calibration(calib, filename="camera_config.json"):
    with open(filename, 'w') as f:
        json.dump(calib, f, indent=4)
    print(f"Calibration saved to {filename}")

def main():
    # Initial WoodScape dictionary structure
    calib = {
        'intrinsic': {
            'width': 1280, 'height': 960, 
            'k1': 600.0, 'k2': 0.0, 'k3': 0.0, 'k4': 0.0, 
            'cx_offset': 0, 'cy_offset': 0, 'aspect_ratio': 1.0
        },
        'extrinsic': {
            'quaternion': [0, 0, 0, 1] 
        }
    }

    cv2.namedWindow("Calibration Tool")
    cv2.createTrackbar("Zoom (k1)", "Calibration Tool", 600, 1500, lambda x: None)
    cv2.createTrackbar("Center X", "Calibration Tool", 500, 1000, lambda x: None) # mapped to offset

    cap = cv2.VideoCapture("../testVideos/shortClip.mp4") # Use your fish video or camera

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Update dict from sliders
        calib['intrinsic']['k1'] = float(cv2.getTrackbarPos("Zoom (k1)", "Calibration Tool"))
        calib['intrinsic']['cx_offset'] = float(cv2.getTrackbarPos("Center X", "Calibration Tool") - 500)

        # Generate the map and remap (using the function from before)
        # map_x, map_y = get_mapping(calib)
        # result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        cv2.imshow("Calibration Tool", frame) # Replace 'frame' with 'result' once mapped
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'): # Press 's' to save the dictionary
            save_calibration(calib)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()