"""Утилиты для работы с изображениями."""

from __future__ import annotations

import cv2
import numpy as np


def crop_roi(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Обрезает область интереса из кадра.

    Args:
        frame: Входной кадр BGR.
        x1, y1, x2, y2: Координаты в пикселях.

    Returns:
        Обрезанный кадр.
    """
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return frame[y1:y2, x1:x2]


def scale_for_ocr(frame: np.ndarray, min_width: int = 300) -> np.ndarray:
    """Масштабирует изображение для лучшего OCR.

    Args:
        frame: Входной кадр.
        min_width: Минимальная ширина в пикселях.

    Returns:
        Масштабированный кадр.
    """
    h, w = frame.shape[:2]
    if w < min_width:
        scale = min_width / w
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return frame
