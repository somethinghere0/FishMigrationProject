from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


def resize_long_side(image: np.ndarray, long_side: Optional[int]) -> np.ndarray:
    if not long_side:
        return image
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def gray_world_white_balance(image_bgr: np.ndarray) -> np.ndarray:
    image = image_bgr.astype(np.float32)
    means = image.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    scale = gray / (means + 1e-6)
    balanced = np.clip(image * scale, 0, 255).astype(np.uint8)
    return balanced


def apply_clahe_lab(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma_correct(image_bgr: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-6:
        return image_bgr
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.array([(i / 255.0) ** inv * 255 for i in np.arange(256)])).astype('uint8')
    return cv2.LUT(image_bgr, table)


def sharpen_unsharp_mask(image_bgr: np.ndarray, amount: float = 1.0, sigma: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
    return cv2.addWeighted(image_bgr, 1.0 + amount, blur, -amount, 0)


def denoise_bilateral(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image_bgr, d=5, sigmaColor=50, sigmaSpace=50)


def undistort_barrel(image_bgr: np.ndarray, camera_matrix: Optional[Sequence], dist_coeffs: Optional[Sequence]) -> np.ndarray:
    if camera_matrix is None or dist_coeffs is None:
        return image_bgr
    K = np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3)
    D = np.asarray(dist_coeffs, dtype=np.float32).reshape(-1, 1)
    h, w = image_bgr.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0, newImgSize=(w, h))
    return cv2.undistort(image_bgr, K, D, None, new_K)


def preprocess_image(image_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    if not cfg.get('enabled', True):
        return image_bgr
    image = image_bgr
    if cfg.get('apply_barrel_undistort', False):
        image = undistort_barrel(image, cfg.get('camera_matrix'), cfg.get('dist_coeffs'))
    if cfg.get('gray_world_white_balance', True):
        image = gray_world_white_balance(image)
    if cfg.get('clahe', True):
        image = apply_clahe_lab(image, cfg.get('clahe_clip_limit', 2.0), cfg.get('clahe_tile_grid_size', 8))
    if cfg.get('gamma_correction', True):
        image = gamma_correct(image, cfg.get('gamma', 1.0))
    if cfg.get('denoise', False):
        image = denoise_bilateral(image)
    if cfg.get('sharpen', False):
        image = sharpen_unsharp_mask(image)
    image = resize_long_side(image, cfg.get('resize_long_side'))
    return image


def preprocess_file(src: Path, dst: Path, cfg: dict) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Could not read image: {src}')
    out = preprocess_image(image, cfg)
    cv2.imwrite(str(dst), out)
