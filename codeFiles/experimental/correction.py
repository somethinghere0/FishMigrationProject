#!/usr/bin/env python3
"""
Improved Circular Fisheye Lens Distortion Correction Script

This script properly corrects extreme circular fisheye distortion using
a polar-to-rectangular remapping approach.
"""

import cv2
import numpy as np
import argparse
import os


def detect_fisheye_circle(frame):
    """
    Detect the circular fisheye region in the frame.
    
    Returns:
        center_x, center_y, radius
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find the circular region
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit a circle to the contour
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        return int(x), int(y), int(radius)
    
    # Fallback
    h, w = frame.shape[:2]
    return w // 2, h // 2, min(w, h) // 2


def create_fisheye_correction_map(src_width, src_height, center_x, center_y, 
                                  radius, strength=1.0):
    """
    Create undistortion maps for fisheye correction.
    Uses proper inverse fisheye model.
    
    Args:
        src_width, src_height: Source image dimensions
        center_x, center_y: Center of fisheye circle
        radius: Radius of fisheye circle
        strength: Correction strength (0.0 to 1.0)
    
    Returns:
        map_x, map_y: Remapping arrays
    """
    map_x = np.zeros((src_height, src_width), dtype=np.float32)
    map_y = np.zeros((src_height, src_width), dtype=np.float32)
    
    # Distortion coefficient - adjust this value to change correction amount
    k1 = -0.3 * strength  # Negative for barrel distortion correction
    
    for y in range(src_height):
        for x in range(src_width):
            # Normalize coordinates relative to center
            dx = (x - center_x) / radius
            dy = (y - center_y) / radius
            
            # Calculate distance from center
            r = np.sqrt(dx**2 + dy**2)
            
            if r > 0 and r <= 1.0:
                # Apply radial distortion correction formula
                # r_corrected = r * (1 + k1*r^2)
                r_corrected = r * (1.0 + k1 * r * r)
                
                # Scale back to pixel coordinates
                scale = r_corrected / r
                
                src_x = center_x + dx * radius * scale
                src_y = center_y + dy * radius * scale
                
                map_x[y, x] = src_x
                map_y[y, x] = src_y
            else:
                # Outside the circle or at center
                map_x[y, x] = x
                map_y[y, x] = y
    
    return map_x, map_y


def create_fisheye_correction_map_v2(src_width, src_height, center_x, center_y, 
                                     radius, strength=1.0, k1=-0.5, k2=-0.3):
    """
    Alternative fisheye correction with more control.
    Uses polynomial distortion model.
    """
    map_x = np.zeros((src_height, src_width), dtype=np.float32)
    map_y = np.zeros((src_height, src_width), dtype=np.float32)
    
    # Apply strength to distortion coefficients
    k1 *= strength
    k2 *= strength
    
    for y in range(src_height):
        for x in range(src_width):
            # Normalize coordinates
            dx = (x - center_x) / radius
            dy = (y - center_y) / radius
            
            r = np.sqrt(dx**2 + dy**2)
            
            if r > 0 and r <= 1.0:
                # Polynomial distortion model
                r2 = r * r
                r4 = r2 * r2
                
                # Correction factor
                distortion = 1.0 + k1 * r2 + k2 * r4
                r_corrected = r * distortion
                
                # Convert back to pixel coordinates
                scale = r_corrected / r if r > 0 else 1.0
                
                src_x = center_x + dx * radius * scale
                src_y = center_y + dy * radius * scale
                
                map_x[y, x] = src_x
                map_y[y, x] = src_y
            else:
                map_x[y, x] = x
                map_y[y, x] = y
    
    return map_x, map_y


def correct_circular_fisheye_video(input_path, output_path, strength=1.0,
                                  k1=-0.5, k2=-0.3, auto_crop=True, 
                                  crop_factor=0.7, show_preview=False):
    """
    Correct circular fisheye distortion in a video file.
    
    Args:
        input_path: Path to input MP4 file
        output_path: Path to output corrected MP4 file
        strength: Overall correction strength (0.0 to 1.0)
        k1: First distortion coefficient (typically -0.3 to -0.7)
        k2: Second distortion coefficient (typically -0.1 to -0.5)
        auto_crop: Crop to remove black borders
        crop_factor: How much to crop (0.5-0.9, lower = more aggressive crop)
        show_preview: Show preview during processing
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height} @ {fps:.2f} fps")
    print(f"Total frames: {total_frames}")
    
    # Read first frame to detect circle
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")
    
    # Detect fisheye circle
    center_x, center_y, radius = detect_fisheye_circle(first_frame)
    print(f"Detected fisheye circle: center=({center_x}, {center_y}), radius={radius}")
    
    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Determine output dimensions
    if auto_crop:
        out_width = int(radius * 2 * crop_factor)
        out_height = int(radius * 2 * crop_factor)
        # Ensure even dimensions
        out_width = out_width if out_width % 2 == 0 else out_width - 1
        out_height = out_height if out_height % 2 == 0 else out_height - 1
        print(f"Output dimensions: {out_width}x{out_height}")
    else:
        out_width = width
        out_height = height
    
    # Create distortion correction maps
    print("Creating distortion correction maps...")
    print(f"Parameters: strength={strength}, k1={k1}, k2={k2}")
    
    map_x, map_y = create_fisheye_correction_map_v2(
        width, height, center_x, center_y, radius, strength, k1, k2
    )
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create output video file: {output_path}")
    
    frame_count = 0
    
    print("\nProcessing video...")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Apply distortion correction
            corrected = cv2.remap(frame, map_x, map_y,
                                 interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
            
            # Crop if requested
            if auto_crop:
                crop_x = (width - out_width) // 2
                crop_y = (height - out_height) // 2
                corrected = corrected[crop_y:crop_y + out_height,
                                    crop_x:crop_x + out_width]
            
            # Write frame
            out.write(corrected)
            
            frame_count += 1
            
            # Progress
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)", 
                      end='\r')
            
            # Preview
            if show_preview:
                display = corrected.copy()
                if out_width > 1280:
                    scale = 1280 / out_width
                    display = cv2.resize(display, None, fx=scale, fy=scale)
                
                cv2.imshow('Corrected Video (Press Q to cancel)', display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing cancelled")
                    break
    
    finally:
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
    
    print(f"\nComplete! Processed {frame_count} frames")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Correct circular fisheye lens distortion in MP4 videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default correction
  python fisheye_fix.py input.mp4 output.mp4
  
  # Stronger correction
  python fisheye_fix.py input.mp4 output.mp4 --k1 -0.7 --k2 -0.5
  
  # Milder correction
  python fisheye_fix.py input.mp4 output.mp4 --k1 -0.3 --k2 -0.2
  
  # Adjust crop amount (lower = more crop, higher = less crop)
  python fisheye_fix.py input.mp4 output.mp4 --crop-factor 0.8
  
  # Keep full frame without cropping
  python fisheye_fix.py input.mp4 output.mp4 --no-crop
  
  # Preview mode
  python fisheye_fix.py input.mp4 output.mp4 --preview
  
  # Fine-tune for your specific lens
  python fisheye_fix.py input.mp4 output.mp4 --k1 -0.6 --k2 -0.4 --strength 0.9
        """
    )
    
    parser.add_argument('input', help='Input MP4 video file')
    parser.add_argument('output', help='Output corrected MP4 video file')
    parser.add_argument('--strength', type=float, default=1.0,
                       help='Overall correction strength (0.0-1.0, default: 1.0)')
    parser.add_argument('--k1', type=float, default=-0.5,
                       help='First distortion coefficient (default: -0.5, try -0.3 to -0.7)')
    parser.add_argument('--k2', type=float, default=-0.3,
                       help='Second distortion coefficient (default: -0.3, try -0.1 to -0.5)')
    parser.add_argument('--no-crop', action='store_true',
                       help='Keep full frame size without cropping')
    parser.add_argument('--crop-factor', type=float, default=0.7,
                       help='Crop factor (0.5-0.9, lower = more aggressive, default: 0.7)')
    parser.add_argument('--preview', action='store_true',
                       help='Show preview window during processing')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if not 0.0 <= args.strength <= 1.0:
        print("Error: Strength must be between 0.0 and 1.0")
        return 1
    
    if not 0.4 <= args.crop_factor <= 1.0:
        print("Error: Crop factor must be between 0.4 and 1.0")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        correct_circular_fisheye_video(
            args.input,
            args.output,
            strength=args.strength,
            k1=args.k1,
            k2=args.k2,
            auto_crop=not args.no_crop,
            crop_factor=args.crop_factor,
            show_preview=args.preview
        )
        return 0
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())