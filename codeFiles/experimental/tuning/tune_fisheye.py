#!/usr/bin/env python3
"""
Interactive Fisheye Distortion Correction Tuner

This script allows you to load a video or image and interactively tune
fisheye distortion parameters to straighten the image and fill the frame.
It uses OpenCV's fisheye model.

Usage:
    python3 tune_fisheye.py <input_file>

Controls:
    - Sliders to adjust Focal Length, Distortion Coefficients (k1-k4), and Zoom.
    - 's': Save current parameters to 'fisheye_params.json' and exit.
    - 'q' or 'Esc': Quit without saving.
"""

import cv2
import numpy as np
import argparse
import json
import sys
import os

def load_media(path):
    """Load the first frame from a video or an image file."""
    if not os.path.exists(path):
        print(f"Error: File {path} not found.")
        sys.exit(1)

    # Try opening as video first
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
        else:
            print("Error: Could not read frame from video.")
    
    # Try opening as image
    frame = cv2.imread(path)
    if frame is None:
        print("Error: Could not load media.")
        sys.exit(1)
    
    return frame

def update_map(val=None):
    """Callback to update the undistortion map and display the frame."""
    global map1, map2, new_K, K, D, DIM, frame, scaled_K
    
    # Get current slider values
    f_scale = cv2.getTrackbarPos('Focal Scale', 'Tuner') / 1000.0
    k1 = (cv2.getTrackbarPos('K1', 'Tuner') - 500) / 100.0
    k2 = (cv2.getTrackbarPos('K2', 'Tuner') - 500) / 100.0
    k3 = (cv2.getTrackbarPos('K3', 'Tuner') - 500) / 100.0
    k4 = (cv2.getTrackbarPos('K4', 'Tuner') - 500) / 100.0
    zoom = cv2.getTrackbarPos('Zoom', 'Tuner') / 100.0
    balance = cv2.getTrackbarPos('Balance', 'Tuner') / 100.0 
    
    # Update Intrinsic Matrix K based on current f_scale
    # Initial estimate: f = width / (2 * tan(fov/2)). 
    # But we simplify: just scale the focal length.
    # Base focal length estimate (assuming ~90 deg FOV for standard, but fisheye is wider)
    # We'll just start with a reasonable guess and let the user tune 'f_scale'.
    h, w = frame.shape[:2]
    K = np.array([[w * f_scale, 0, w/2],
                  [0, w * f_scale, h/2],
                  [0, 0, 1]], dtype=np.float32)
    
    D = np.array([k1, k2, k3, k4], dtype=np.float32)
    
    # The 'balance' parameter in estimateNewCameraMatrixForUndistortRectify is for normal lens model.
    # cv2.fisheye doesn't have a direct 'balance' parameter in the same way, 
    # but we can simulate zoom/crop by modifying the new Camera Matrix.
    
    # We can use cv2.fisheye.estimateNewCameraMatrixForUndistortRectify if we had a calibration.
    # Since we are "blindly" tuning, we will construct the new K manually.
    
    # Apply zoom to the new K
    nk = K.copy()
    nk[0, 0] *= zoom
    nk[1, 1] *= zoom
    
    # Optimization: Only recompute maps if parameters changed (not implemented for simplicity here)
    try:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), nk, (w, h), cv2.CV_16SC2)
        undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Resize for display if too large
        disp_h, disp_w = undistorted.shape[:2]
        if disp_w > 1280:
            scale = 1280 / disp_w
            display = cv2.resize(undistorted, (int(disp_w*scale), int(disp_h*scale)))
        else:
            display = undistorted
            
        cv2.imshow('Tuner', display)
        
    except cv2.error as e:
        # Invalid parameters can cause errors
        pass

def main():
    global frame, K, D, DIM, map1, map2
    
    parser = argparse.ArgumentParser(description='Interactive Fisheye Tuner')
    parser.add_argument('input_file', help='Path to video or image file')
    args = parser.parse_args()
    
    frame = load_media(args.input_file)
    h, w = frame.shape[:2]
    DIM = (w, h)
    
    cv2.namedWindow('Tuner')
    
    # Initial parameters
    # Focal Scale: 1000 -> 1.0. Range 0.1 to 5.0
    cv2.createTrackbar('Focal Scale', 'Tuner', 500, 2000, update_map) # Default 0.5
    
    # Distortion Coeffs: 500 -> 0.0. Range -5.0 to 5.0
    cv2.createTrackbar('K1', 'Tuner', 500, 1000, update_map)
    cv2.createTrackbar('K2', 'Tuner', 500, 1000, update_map)
    cv2.createTrackbar('K3', 'Tuner', 500, 1000, update_map)
    cv2.createTrackbar('K4', 'Tuner', 500, 1000, update_map)
    
    # Zoom/Scale: 100 -> 1.0. Range 0.1 to 3.0
    cv2.createTrackbar('Zoom', 'Tuner', 100, 300, update_map)

    # Note: K1 is usually negative for fisheye (-0.1 to -1.0)
    # Set some reasonable defaults for fisheye
    cv2.setTrackbarPos('K1', 'Tuner', 400) # -1.0
    cv2.setTrackbarPos('K2', 'Tuner', 500) # 0.0
    
    update_map()
    
    print("\n" + "="*50)
    print("INSTRUCTIONS:")
    print("1. Click on the 'Tuner' window to focus it.")
    print("2. Adjust sliders to correct distortion.")
    print("3. Press 's' WHILE THE WINDOW IS FOCUSED to save parameters.")
    print("4. Press 'q' WHILE THE WINDOW IS FOCUSED to quit.")
    print("="*50 + "\n")
    
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            # Save parameters
            f_scale = cv2.getTrackbarPos('Focal Scale', 'Tuner') / 1000.0
            k1 = (cv2.getTrackbarPos('K1', 'Tuner') - 500) / 100.0
            k2 = (cv2.getTrackbarPos('K2', 'Tuner') - 500) / 100.0
            k3 = (cv2.getTrackbarPos('K3', 'Tuner') - 500) / 100.0
            k4 = (cv2.getTrackbarPos('K4', 'Tuner') - 500) / 100.0
            zoom = cv2.getTrackbarPos('Zoom', 'Tuner') / 100.0
            
            params = {
                'f_scale': f_scale,
                'k1': k1,
                'k2': k2,
                'k3': k3,
                'k4': k4,
                'zoom': zoom,
                'resolution': [w, h]
            }
            
            with open('fisheye_params.json', 'w') as f:
                json.dump(params, f, indent=4)
            print("Parameters saved to fisheye_params.json")
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
