from __future__ import annotations
import cv2
import numpy as np


def gray_world_white_balance_bgr(img: np.ndarray) -> np.ndarray:
    """Simple gray-world white balance for underwater color casts."""
    img_f = img.astype(np.float32)
    means = img_f.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    scale = gray / (means + 1e-6)
    out = img_f * scale
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_clahe_bgr(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """CLAHE on L channel in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def gamma_correct(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if gamma is None or abs(gamma - 1.0) < 1e-6:
        return img
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def sharpen_bgr(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def preprocess_frame_bgr(frame: np.ndarray, cfg: dict) -> np.ndarray:
    if not cfg or not cfg.get("enabled", True):
        return frame
    out = frame.copy()
    if cfg.get("gray_world_white_balance", False):
        out = gray_world_white_balance_bgr(out)
    if cfg.get("clahe", False):
        out = apply_clahe_bgr(out)
    if cfg.get("gamma", None) is not None:
        out = gamma_correct(out, float(cfg.get("gamma", 1.0)))
    if cfg.get("denoise", False):
        out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)
    if cfg.get("sharpen", False):
        out = sharpen_bgr(out)
    return out
