import cv2
import numpy as np
from pyquaternion import Quaternion

def get_mapping(calib, hfov=np.deg2rad(190), vfov=np.deg2rad(143), out_shape=(720, 1280)):
    """
    Computes the pixel mapping to transform circular fisheye to a flat cylindrical view.
    """
    h_out, w_out = out_shape
    
    # --- 1. Extrinsic Rotation Logic ---
    q = calib['extrinsic']['quaternion']
    # Create rotation matrix: WoodScape/OmniDet format [x, y, z, w]
    R_raw = Quaternion(w=q[3], x=q[0], y=q[1], z=q[2]).rotation_matrix.T
    
    # Coordinate system transformation: RDF to FLU
    rdf_to_flu = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
    R = R_raw @ rdf_to_flu 

    # Azimuth and Tilt (correcting for camera mounting angle)
    azimuth = np.arccos(R[2, 2] / np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2))
    if R[0, 2] < 0: azimuth = 2*np.pi - azimuth
    tilt = -np.arccos(np.sqrt(R[0, 2]**2 + R[2, 2]**2))

    # Level the forward axis to the ground
    Ry = np.array([[np.cos(azimuth), 0, np.sin(azimuth)],
                     [0, 1, 0],
                     [-np.sin(azimuth), 0, np.cos(azimuth)]]).T
    R_final = R @ Ry 

    # --- 2. Intrinsic Setup ---
    f = calib['intrinsic']['k1']
    # Intrinsic matrix for the target cylindrical projection
    K = np.array([[f, 0, w_out/2],
                  [0, f, f * np.tan(vfov/2 + tilt)],
                  [0, 0, 1]], dtype=np.float32)
    K_inv = np.linalg.inv(K)

    # --- 3. Ray Generation ---
    xv, yv = np.meshgrid(np.arange(w_out), np.arange(h_out))
    # Create homogeneous pixel coordinates (H, W, 3)
    p = np.stack([xv, yv, np.ones_like(xv)], axis=-1)
    
    # Back-project pixels to cylindrical rays
    # (H, W, 3) dot (3, 3)
    r_cyl = p @ K_inv.T
    r_cyl /= np.linalg.norm(r_cyl[:, :, [2]], axis=2, keepdims=True) # Unit radius
    
    # Convert Cylindrical to Cartesian
    x_cart = np.sin(r_cyl[:, :, 0])
    y_cart = r_cyl[:, :, 1]
    z_cart = np.cos(r_cyl[:, :, 0])
    
    # Stack to (H, W, 3)
    r_cart = np.stack([x_cart, y_cart, z_cart], axis=-1)
    
    # Apply rotation: (H, W, 3) @ (3, 3)
    r_rotated = r_cart @ R_final.T

    # --- 4. Project onto Fisheye (Polynomial Model) ---
    # Angle from optical axis
    norm_r = np.linalg.norm(r_rotated, axis=2)
    theta = np.arccos(np.clip(r_rotated[:, :, 2] / norm_r, -1.0, 1.0))
    
    # WoodScape Polynomial: r = k1*theta + k2*theta^2 + ...
    k = [calib['intrinsic'].get(f'k{i}', 0.0) for i in range(1, 5)]
    rho = k[0]*theta + k[1]*theta**2 + k[2]*theta**3 + k[3]*theta**4
    
    # Project back to 2D image coordinates
    chi = np.linalg.norm(r_rotated[:, :, :2], axis=2)
    # Avoid division by zero at the center
    chi_mask = (chi != 0)
    
    u = np.zeros_like(chi)
    v = np.zeros_like(chi)
    u[chi_mask] = (rho[chi_mask] * r_rotated[chi_mask, 0]) / chi[chi_mask]
    v[chi_mask] = (rho[chi_mask] * r_rotated[chi_mask, 1]) / chi[chi_mask]

    c_X = calib['intrinsic']['cx_offset'] + calib['intrinsic']['width'] / 2 - 0.5
    c_Y = calib['intrinsic']['cy_offset'] + calib['intrinsic']['height'] / 2 - 0.5
    
    mapx = u + c_X
    mapy = v * calib['intrinsic']['aspect_ratio'] + c_Y
    
    return mapx.astype(np.float32), mapy.astype(np.float32)

def main(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Base Calibration Parameters
    # Adjust k1 (Zoom) to fill your specific screen
    calib = {
        'intrinsic': {
            'width': 1280, 'height': 960, 
            'k1': 600.0, 'k2': 0.0, 'k3': 0.0, 'k4': 0.0, 
            'cx_offset': 0, 'cy_offset': 0, 'aspect_ratio': 1.0
        },
        'extrinsic': {
            'quaternion': [0, 0, 0, 1] # [x, y, z, w] Identity
        }
    }

    cv2.namedWindow("Fisheye Corrector", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Zoom (k1)", "Fisheye Corrector", 600, 2000, lambda x: None)
    
    print("Controls: 'q' to Quit, 's' to Save Frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop
            continue

        # Get current Zoom from trackbar
        calib['intrinsic']['k1'] = float(cv2.getTrackbarPos("Zoom (k1)", "Fisheye Corrector"))

        # Generate maps (720p output)
        mx, my = get_mapping(calib, out_shape=(720, 1280))

        # Remap
        undistorted = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        cv2.imshow("Fisheye Corrector", undistorted)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("corrected_frame.jpg", undistorted)
            print("Frame saved as corrected_frame.jpg")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change 'your_video.mp4' to your file path
    main("testVideos/shortClip.mp4") # 0 uses your webcam, or replace with "video.mp4"

