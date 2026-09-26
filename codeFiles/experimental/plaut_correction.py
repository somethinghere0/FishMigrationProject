import cv2
import numpy as np
from pyquaternion import Quaternion


def get_mapping(calib, hfov=np.deg2rad(190), vfov=np.deg2rad(143)):
    """
    Compute the pixel mapping from a fisheye image to a cylindrical image
    :param calib: calibration in WoodScape format, as a dictionary
    :param hfov: horizontal field of view, in radians
    :param vfov: vertical field of view, in radians
    :return: horizontal and vertical mapping
    """
    # Prepare intrinsic and extrinsic matrices for the cylindrical image
    R = Quaternion(w=calib['extrinsic']['quaternion'][3],
                   x=calib['extrinsic']['quaternion'][0],
                   y=calib['extrinsic']['quaternion'][1],
                   z=calib['extrinsic']['quaternion'][2]).rotation_matrix.T
    rdf_to_flu = np.array([[0, 0, 1],
                           [-1, 0, 0],
                           [0, -1, 0]], dtype=np.float64)
    R = R @ rdf_to_flu  # Rotation from vehicle to camera includes FLU-to-RDF. Remove FLU-to-RDF from R.
    azimuth = np.arccos(R[2, 2] / np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2))  # azimuth angle parallel to the ground
    if R[0, 2] < 0:
        azimuth = 2*np.pi - azimuth
    tilt = -np.arccos(np.sqrt(R[0, 2]**2 + R[2, 2]**2))  # elevation to the ground plane
    Ry = np.array([[np.cos(azimuth), 0, np.sin(azimuth)],
                     [0, 1, 0],
                     [-np.sin(azimuth), 0, np.cos(azimuth)]]).T
    R = R @ Ry  # now forward axis is parallel to the ground, but in the direction of the camera (not vehicle's forward)
    f = calib['intrinsic']['k1']
    h, w = int(2*f*np.tan(vfov/2)), int(f*hfov)  # cylindrical image has a different size than the fisheye image
    K = np.array([[f, 0, w/2],
                  [0, f, f * np.tan(vfov/2 + tilt)],
                  [0, 0, 1]], dtype=np.float32)  # intrinsic matrix for the cylindrical projection
    K_inv = np.linalg.inv(K)
    # Create pixel grid and compute a ray for every pixel
    xv, yv = np.meshgrid(range(w), range(h), indexing='xy')
    p = np.stack([xv, yv, np.ones_like(xv)])  # pixel homogeneous coordinates
    p = p.transpose(1, 2, 0)[:, :, :, np.newaxis]
    r = K_inv @ p  # r is in cylindrical coordinates
    r /= r[:, :, [2], :]  # r is now in cylindrical coordinates with unit cylindrical radius
    # Convert to Cartesian coordinates
    r[:, :, 2, :] = np.cos(r[:, :, 0, :])
    r[:, :, 0, :] = np.sin(r[:, :, 0, :])
    r[:, :, 1, :] = r[:, :, 1, :]
    r = R @ r  # extrinsic rotation from an upright cylinder to the camera axis
    theta = np.arccos(r[:, :, [2], :] / np.linalg.norm(r, axis=2, keepdims=True))  # compute incident angle
    # project the ray onto the fisheye image according to the fisheye model and intrinsic calibration
    c_X = calib['intrinsic']['cx_offset'] + calib['intrinsic']['width'] / 2 - 0.5
    c_Y = calib['intrinsic']['cy_offset'] + calib['intrinsic']['height'] / 2 - 0.5
    k1, k2, k3, k4 = [calib['intrinsic']['k%d' % i] for i in range(1, 5)]
    rho = k1 * theta + k2 * theta ** 2 + k3 * theta ** 3 + k4 * theta ** 4
    chi = np.linalg.norm(r[:, :, :2, :], axis=2, keepdims=True)
    u = np.true_divide(rho * r[:, :, [0], :], chi, out=np.zeros_like(chi), where=(chi != 0))  # horizontal
    v = np.true_divide(rho * r[:, :, [1], :], chi, out=np.zeros_like(chi), where=(chi != 0))  # vertical
    mapx = u[:, :, 0, 0] + c_X
    mapy = v[:, :, 0, 0] * calib['intrinsic']['aspect_ratio'] + c_Y
    return mapx, mapy


def fisheye_to_cylindrical(image, calib):
    """
    Warp a fisheye image to a cylindrical image
    :param image: fisheye image, as a numpy array
    :param calib: calibration in WoodScape format, as a dictionary
    :return: cylindrical image
    """
    mapx, mapy = get_mapping(calib)
    return cv2.remap(image, mapx.astype(np.float32), mapy.astype(np.float32), interpolation=cv2.INTER_LINEAR)