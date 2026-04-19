"""Модуль загрузки и базовой обработки видеозаписей.

Предоставляет утилиты для:
- Генерации уникальных ID видео
- Получения метаданных видеофайлов
- Определения типа записи (direct / handheld)
"""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np

from app.models.schemas import VideoType


def generate_video_id() -> str:
    """Генерирует уникальный ID для видео.

    Returns:
        Строка UUID для идентификации видео.
    """
    return str(uuid.uuid4())


def get_video_info(video_path: str | Path) -> dict:
    """Возвращает базовую информацию о видеофайле.

    Args:
        video_path: Путь к видеофайлу.

    Returns:
        Словарь с метаданными видео:
        - resolution: кортеж (width, height)
        - fps: кадров в секунду
        - total_frames: общее количество кадров
        - duration_s: длительность в секундах

    Raises:
        ValueError: Если не удалось открыть видеофайл.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0.0

    cap.release()

    return {
        "resolution": (width, height),
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": duration_s,
    }


def detect_video_type(
    frame: np.ndarray,
    prev_frame: np.ndarray | None = None,
    video_path: str | None = None,
) -> str:
    """Определяет тип видеозаписи: direct / handheld / handheld_angle.

    Алгоритм анализирует яркость границ и соотношение сторон
    для определения типа записи SCADA-экрана.

    Args:
        frame: Текущий кадр в формате BGR.
        prev_frame: Предыдущий кадр (опционально, для анализа стабильности).
        video_path: Путь к видео (опционально).

    Returns:
        Строка с типом видео: "direct", "handheld", или "handheld_angle".
    """
    if frame is None or frame.size == 0:
        return "direct"

    height, width = frame.shape[:2]

    # Анализ границ изображения
    # Тёмные границы характерны для handheld съёмки
    border_ratio = _analyze_borders(frame)

    # Проверка соотношения сторон
    aspect_ratio = width / height if height > 0 else 1.0

    # Классификация
    if border_ratio > 0.15:
        # Значительные тёмные границы — handheld
        if aspect_ratio < 1.6:
            return "handheld_angle"
        return "handheld"

    # Стабильное соотношение сторон без границ — direct capture
    return "direct"


def _analyze_borders(frame: np.ndarray) -> float:
    """Анализирует тёмные границы кадра.

    Args:
        frame: Кадр в формате BGR.

    Returns:
        Отношение площади тёмных границ к общей площади кадра.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Порог для определения тёмных пикселей
    dark_threshold = 30
    dark_mask = gray < dark_threshold

    # Анализируем полосы по краям (10% от размера)
    border_width = width // 10
    border_height = height // 10

    top_border = dark_mask[:border_height, :]
    bottom_border = dark_mask[-border_height:, :]
    left_border = dark_mask[:, :border_width]
    right_border = dark_mask[:, -border_width:]

    # Считаем долю тёмных пикселей в границах
    total_border_pixels = (
        top_border.size + bottom_border.size +
        left_border.size + right_border.size
    )
    dark_border_pixels = (
        top_border.sum() + bottom_border.sum() +
        left_border.sum() + right_border.sum()
    )

    return dark_border_pixels / total_border_pixels if total_border_pixels > 0 else 0.0


# ---------------------------------------------------------------------------
# Функции для обратной совместимости (deprecated, но могут использоваться)
# ---------------------------------------------------------------------------


def _ms_to_timestamp(ms: float) -> str:
    """Конвертирует миллисекунды в формат HH:MM:SS.mmm.

    Args:
        ms: Время в миллисекундах.

    Returns:
        Строка в формате HH:MM:SS.mmm.
    """
    total_seconds = int(ms // 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int(ms % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def extract_frames(
    video_path: str | Path,
    interval_ms: int = 500,
    max_frames: int | None = None,
) -> list[tuple[np.ndarray, str]]:
    """Извлекает кадры из видео с заданным интервалом.

    Предупреждение: Загружает все кадры в память!
    Для больших видео используйте LazyFrameExtractor.

    Args:
        video_path: Путь к видеофайлу.
        interval_ms: Интервал между кадрами в миллисекундах.
        max_frames: Максимальное количество кадров (None = без ограничений).

    Returns:
        Список пар (кадр, timestamp).

    Raises:
        ValueError: Если не удалось открыть видео.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = max(1, int(fps * interval_ms / 1000.0))

    frames: list[tuple[np.ndarray, str]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval_frames == 0:
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = _ms_to_timestamp(timestamp_ms)
            frames.append((frame, ts))

            if max_frames is not None and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames


class LazyFrameExtractor:
    """Ленивый извлекатель кадров — загружает кадры по требованию.

    Экономит память при обработке больших видео.
    """

    def __init__(
        self,
        video_path: str | Path,
        interval_ms: int = 500,
        max_frames: int | None = None,
    ) -> None:
        self._video_path = str(video_path)
        self._interval_ms = interval_ms
        self._max_frames = max_frames
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 25.0
        self._interval_frames: int = 1
        self._frame_idx: int = 0
        self._extracted_count: int = 0
        self._total_frames_video: int = 0
        self._width: int = 0
        self._height: int = 0

    @property
    def total_frames(self) -> int:
        """Оценка общего количества извлекаемых кадров."""
        if self._total_frames_video <= 0:
            return 0
        estimated = self._total_frames_video // self._interval_frames
        if self._max_frames is not None:
            return min(estimated, self._max_frames)
        return estimated

    def __enter__(self) -> "LazyFrameExtractor":
        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {self._video_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 25.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames_video = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._interval_frames = max(1, int(self._fps * self._interval_ms / 1000.0))
        self._frame_idx = 0
        self._extracted_count = 0

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> "LazyFrameExtractor":
        return self

    def __next__(self) -> tuple[np.ndarray, float]:
        if self._cap is None:
            raise StopIteration

        while True:
            ret, frame = self._cap.read()
            if not ret:
                raise StopIteration

            current_idx = self._frame_idx
            self._frame_idx += 1

            if current_idx % self._interval_frames == 0:
                timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                self._extracted_count += 1

                if self._max_frames is not None and self._extracted_count > self._max_frames:
                    raise StopIteration

                return frame, timestamp_ms
