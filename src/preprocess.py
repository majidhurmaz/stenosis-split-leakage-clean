"""Image pre-processing shared by data preparation, training and real-time inference.

Implements the pre-processing steps of the proposal:
  * grayscale conversion / normalisation
  * CLAHE contrast enhancement (limits noise amplification of plain AHE)
  * optional light denoising
  * DICOM cine reading (for the MTI HMC Peshawar data) incl. pixel-spacing calibration
"""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def normalise_uint8(img: np.ndarray) -> np.ndarray:
    """Min-max normalise any dtype (e.g. 12/16-bit DICOM) to uint8 0-255."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)


def enhance(img: np.ndarray, clahe: bool = True, clip: float = 2.0, tile: int = 8,
            denoise: bool = False) -> np.ndarray:
    """Return a 3-channel uint8 image ready for the detector."""
    g = to_gray(img)
    if g.dtype != np.uint8:
        g = normalise_uint8(g)
    if denoise:
        g = cv2.medianBlur(g, 3)
    if clahe:
        g = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(g)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def read_dicom_cine(path: str):
    """Read a (multi-frame) angiography DICOM.

    Returns (frames[N,H,W] uint8, mm_per_pixel or None, fps or None).
    mm_per_pixel comes from PixelSpacing / ImagerPixelSpacing (corrected by the
    estimated radiographic magnification factor when present).
    """
    import pydicom  # optional dependency, only needed for DICOM input

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim == 4:  # RGB cine
        arr = arr[..., 0]
    frames = np.stack([normalise_uint8(f) for f in arr])

    mmpp = None
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        if tag in ds:
            mmpp = float(ds[tag].value[0])
            break
    if mmpp is not None and "EstimatedRadiographicMagnificationFactor" in ds:
        mmpp /= float(ds.EstimatedRadiographicMagnificationFactor)

    fps = None
    if "CineRate" in ds:
        fps = float(ds.CineRate)
    elif "FrameTime" in ds:
        fps = 1000.0 / float(ds.FrameTime)
    return frames, mmpp, fps
