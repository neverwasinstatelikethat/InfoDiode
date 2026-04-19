"""Предобработчик кадров для OCR.

Три pipeline обработки в зависимости от типа видео:
A) Direct screen capture — минимальная обработка
B) Handheld camera — стабилизация + коррекция перспективы
C) Phone at angle — полная обработка + дешифтинг + резкость
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from app.core.screen_detector import (
    apply_homography,
    compute_homography,
    detect_screen,
    stabilize_frame,
)
from app.models.schemas import VideoType

logger = logging.getLogger(__name__)

# Максимальный размер кадра для OCR (по длинной стороне)
# PaddleOCR работает быстрее на меньших кадрах, качество не страдает до ~1280px
# ДОЛЖЕН совпадать с MAX_OCR_DIM из ocr_pipeline.py
MAX_OCR_DIM = 1280

# Кэшированный CLAHE объект для повторного использования
# Создаётся один раз при первом вызове, переиспользуется между кадрами
_clahe_instance: cv2.CLAHE | None = None

# Кэш гомографии для handheld видео — матрица не меняется между кадрами
# одного видео (при условии что камера неподвижна)
_homography_cache: dict[str, np.ndarray] = {}

# Кэш углов экрана — определяется один раз для видео
_screen_corners_cache: dict[str, np.ndarray] = {}

# Эпоха кэша — инкрементируется при смене вкладки/сцены для инвалидации кэша
# Включается в ключ кэша: f"{epoch}_{video_id}_{w}x{h}"
_cache_epoch: int = 0


def get_cache_epoch() -> int:
    """Возвращает текущую эпоху кэша.

    Returns:
        Текущее значение эпохи кэша.
    """
    return _cache_epoch


def increment_cache_epoch() -> int:
    """Инкрементирует эпоху кэша для инвалидации всех кэшей.

    Вызывать при смене вкладки SCADA или мнемосхемы.
    Возвращает новое значение эпохи.

    Returns:
        Новое значение эпохи кэша.
    """
    global _cache_epoch
    _cache_epoch += 1
    logger.info("Эпоха кэша инкрементирована: %d", _cache_epoch)
    return _cache_epoch


def _get_clahe() -> cv2.CLAHE:
    """Возвращает кэшированный CLAHE объект.

    Returns:
        Экземпляр CLAHE с clipLimit=2.0 и tileGridSize=(8, 8).
    """
    global _clahe_instance
    if _clahe_instance is None:
        _clahe_instance = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _clahe_instance


def clear_preprocess_cache() -> None:
    """Сбрасывает кэш предобработки (вызывать при смене видео или сцены).

    Также инкрементирует эпоху кэша для инвалидации ключей.
    """
    global _homography_cache, _screen_corners_cache, _cache_epoch
    _homography_cache.clear()
    _screen_corners_cache.clear()
    _cache_epoch += 1
    logger.debug("Кэш предобработки сброшен, эпоха: %d", _cache_epoch)


def resize_for_ocr(frame: np.ndarray, max_dim: int = MAX_OCR_DIM) -> tuple[np.ndarray, float]:
    """Уменьшает кадр до max_dim по длинной стороне для ускорения обработки.

    Вызывается ОДИН РАЗ в начале preprocess_frame. Все последующие операции
    (CLAHE, стабилизация, гомография) работают на уменьшенном кадре.

    Args:
        frame: Входной кадр BGR.
        max_dim: Максимальный размер по длинной стороне.

    Returns:
        (resized_frame, scale_factor) — масштабированный кадр и коэффициент.
    """
    h, w = frame.shape[:2]
    max_side = max(h, w)
    if max_side <= max_dim:
        return frame, 1.0
    scale = max_dim / max_side
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    logger.debug("Кадр уменьшен: %dx%d -> %dx%d (scale=%.2f)", w, h, new_w, new_h, scale)
    return resized, scale


def preprocess_frame(
    frame: np.ndarray,
    video_type: VideoType,
    prev_frame: np.ndarray | None = None,
    screen_corners: np.ndarray | None = None,
    video_id: str = "",
    resize_first: bool = True,
) -> tuple[np.ndarray, float]:
    """Предобрабатывает кадр в зависимости от типа видео.

    Оптимизация: уменьшает кадр до MAX_OCR_DIM в начале обработки,
    чтобы все последующие операции работали на меньшем кадре.

    Args:
        frame: Входной кадр BGR.
        video_type: Тип видео (direct / handheld / handheld_angle).
        prev_frame: Предыдущий кадр для стабилизации.
        screen_corners: Углы экрана (если уже известны).
        video_id: ID видео для кэширования предобработки.
        resize_first: Если True, уменьшает кадр до MAX_OCR_DIM перед обработкой.

    Returns:
        (preprocessed_frame, scale_factor) — предобработанный кадр и коэффициент масштабирования.
        Коэффициент нужен для обратного масштабирования координат OCR.
    """
    t0 = time.perf_counter()
    original_h, original_w = frame.shape[:2]

    # ОПТИМИЗАЦИЯ: уменьшаем кадр в начале обработки
    # Все последующие операции (CLAHE, стабилизация, гомография)
    # будут работать на уменьшенном кадре — это значительно быстрее
    scale = 1.0
    if resize_first:
        frame, scale = resize_for_ocr(frame)

    h, w = frame.shape[:2]

    match video_type:
        case VideoType.DIRECT:
            result = _pipeline_direct(frame)
        case VideoType.HANDHELD:
            result = _pipeline_handheld(frame, prev_frame, screen_corners, video_id)
        case VideoType.HANDHELD_ANGLE:
            result = _pipeline_handheld_angle(frame, prev_frame, screen_corners, video_id)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.debug(
        "Предобработка кадра %dx%d -> %dx%d (%s, scale=%.2f): %.1f мс",
        original_w, original_h, w, h, video_type.value, scale, elapsed_ms
    )

    return result, scale


def _pipeline_direct(frame: np.ndarray) -> np.ndarray:
    """Pipeline A: прямая захват экрана.

    Минимальная обработка: CLAHE + масштабирование мелкого текста.

    Args:
        frame: Кадр BGR.

    Returns:
        Предобработанный кадр.
    """
    # CLAHE для улучшения контраста
    result = _apply_clahe(frame)
    return result


def _pipeline_handheld(
    frame: np.ndarray,
    prev_frame: np.ndarray | None = None,
    screen_corners: np.ndarray | None = None,
    video_id: str = "",
) -> np.ndarray:
    """Pipeline B: handheld камера (телефон прямо).

    Стабилизация + коррекция перспективы + CLAHE.
    Оптимизация: кэширует гомографию и углы экрана между кадрами.

    Args:
        frame: Кадр BGR.
        prev_frame: Предыдущий кадр.
        screen_corners: Углы экрана.
        video_id: ID видео для ключа кэша.

    Returns:
        Предобработанный кадр.
    """
    # Стабилизация
    if prev_frame is not None:
        frame = stabilize_frame(prev_frame, frame)

    # Детекция экрана и коррекция перспективы (с кэшированием)
    # Ключ кэша включает эпоху для инвалидации при смене вкладки/сцены
    epoch = get_cache_epoch()
    corners_cache_key = f"{epoch}_{video_id}"

    if screen_corners is not None:
        corners = screen_corners
    elif video_id and corners_cache_key in _screen_corners_cache:
        corners = _screen_corners_cache[corners_cache_key]
    else:
        corners = detect_screen(frame)
        if corners is not None and video_id:
            _screen_corners_cache[corners_cache_key] = corners

    if corners is not None:
        h, w = frame.shape[:2]
        # Кэшируем гомографию — она не меняется между кадрами одного видео
        # Ключ включает эпоху для инвалидации при смене вкладки/сцены
        homography_cache_key = f"{epoch}_{video_id}_{w}x{h}"
        if homography_cache_key in _homography_cache:
            homography = _homography_cache[homography_cache_key]
        else:
            homography = compute_homography(corners, (w, h))
            if video_id:
                _homography_cache[homography_cache_key] = homography
        frame = apply_homography(frame, homography, (w, h))

    # CLAHE
    return _apply_clahe(frame)


def _pipeline_handheld_angle(
    frame: np.ndarray,
    prev_frame: np.ndarray | None = None,
    screen_corners: np.ndarray | None = None,
    video_id: str = "",
) -> np.ndarray:
    """Pipeline C: телефон под углом.

    Полная обработка: стабилизация + коррекция перспективы +
    резкость + адаптивный порог + коррекция дисторсии.
    Оптимизация: наследует кэширование гомографии от Pipeline B.

    Args:
        frame: Кадр BGR.
        prev_frame: Предыдущий кадр.
        screen_corners: Углы экрана.
        video_id: ID видео для ключа кэша.

    Returns:
        Предобработанный кадр.
    """
    # Все шаги Pipeline B (с кэшированием)
    result = _pipeline_handheld(frame, prev_frame, screen_corners, video_id)

    # Дополнительная резкость после dewarping
    result = _sharpen(result)

    # Коррекция дисторсии (баррель/пинкушон)
    result = _correct_distortion(result)

    return result


def _apply_clahe(frame: np.ndarray) -> np.ndarray:
    """Применяет CLAHE (адаптивная эквализация гистограммы).

    Использует кэшированный CLAHE объект для оптимизации производительности.

    Args:
        frame: Кадр BGR.

    Returns:
        Кадр с улучшенным контрастом.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = _get_clahe()  # Используем кэшированный объект
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _sharpen(frame: np.ndarray) -> np.ndarray:
    """Повышает резкость кадра (необходима после dewarping).

    Args:
        frame: Кадр BGR.

    Returns:
        Резкий кадр.
    """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    return cv2.filter2D(frame, -1, kernel)


def _correct_distortion(frame: np.ndarray) -> np.ndarray:
    """Корректирует баррель/пинкушон дисторсию.

    Args:
        frame: Кадр BGR.

    Returns:
        Кадр с исправленной дисторсией.
    """
    h, w = frame.shape[:2]

    # Стандартные параметры дисторсии для телефонной камеры
    k1 = -0.1  # Баррель-дисторсия
    p1 = 0.0
    p2 = 0.0
    k2 = 0.0
    k3 = 0.0

    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    cam_matrix = np.array(
        [[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]],
        dtype=np.float64,
    )

    return cv2.undistort(frame, cam_matrix, dist_coeffs)
