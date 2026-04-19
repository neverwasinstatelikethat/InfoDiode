"""Модуль загрузки и предобработки видеозаписей.

Извлечение кадров с интервалом 500мс, определение типа видео
(direct / handheld / handheld_angle), базовая информация о видео.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np

from app.models.schemas import VideoType

# Кэш для результатов определения типа видео (ключ = хэш кадра)
_video_type_cache: dict[str, VideoType] = {}


def clear_video_type_cache() -> None:
    """Очищает кэш определённых типов видео.

    Используется в тестах для сброса состояния между тестами.
    """
    global _video_type_cache
    _video_type_cache.clear()


def extract_frames(
    video_path: str | Path,
    interval_ms: int = 500,
    max_frames: int | None = None,
) -> list[tuple[np.ndarray, str]]:
    """Извлекает кадры из видео с заданным интервалом.

    Args:
        video_path: Путь к видеофайлу.
        interval_ms: Интервал между кадрами в миллисекундах.
        max_frames: Максимальное количество кадров для извлечения (None = без ограничений).

    Returns:
        Список пар (кадр, таймстемп HH:MM:SS.mmm).
    """
    import logging

    logger = logging.getLogger(__name__)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval_frames = int(fps * interval_ms / 1000.0)
    if interval_frames < 1:
        interval_frames = 1

    # Оценка количества кадров и памяти
    estimated_frames = total_frames_video // interval_frames if total_frames_video > 0 else 0
    if max_frames is not None:
        estimated_frames = min(estimated_frames, max_frames)

    # Примерный расчёт памяти: кадр ~ width * height * 3 байт
    bytes_per_frame = width * height * 3
    estimated_memory_mb = (estimated_frames * bytes_per_frame) / (1024 * 1024)

    logger.info(
        "Извлечение кадров: %dx%d, FPS=%.1f, интервал=%d кадров, "
        "ожидается %d кадров, ~%.1f МБ памяти",
        width, height, fps, interval_frames, estimated_frames, estimated_memory_mb
    )

    frames: list[tuple[np.ndarray, str]] = []
    frame_idx = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval_frames == 0:
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = _ms_to_timestamp(timestamp_ms)
            frames.append((frame, ts))
            extracted_count += 1

            # Проверяем ограничение по количеству кадров
            if max_frames is not None and extracted_count >= max_frames:
                logger.info("Достигнут лимит max_frames=%d, остановка извлечения", max_frames)
                break

        frame_idx += 1

    cap.release()

    actual_memory_mb = (len(frames) * bytes_per_frame) / (1024 * 1024)
    logger.info(
        "Извлечено %d кадров, фактическое использование памяти: ~%.1f МБ",
        len(frames), actual_memory_mb
    )

    return frames


class LazyFrameExtractor:
    """Ленивый извлекатель кадров — загружает кадры по требованию.

    Вместо загрузки ВСЕХ кадров в память (как extract_frames()),
    этот класс открывает VideoCapture и извлекает кадры по одному,
    что экономит ~4 ГБ памяти для типичного видео.

    Поддерживает контекстный менеджер для автоматического освобождения ресурса.

    Usage:
        with LazyFrameExtractor(video_path, interval_ms=500) as extractor:
            for frame, timestamp_ms in extractor:
                process(frame)
    """

    def __init__(
        self,
        video_path: str | Path,
        interval_ms: int = 500,
        max_frames: int | None = None,
    ) -> None:
        """Инициализирует ленивый извлекатель.

        Args:
            video_path: Путь к видеофайлу.
            interval_ms: Интервал между кадрами в миллисекундах.
            max_frames: Максимальное количество кадров (None = без ограничений).
        """
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

    @property
    def video_info(self) -> dict:
        """Возвращает информацию о видео."""
        return {
            "resolution": (self._width, self._height),
            "fps": self._fps,
            "total_frames_video": self._total_frames_video,
            "interval_frames": self._interval_frames,
            "estimated_extracted": self.total_frames,
        }

    def __enter__(self) -> LazyFrameExtractor:
        """Открывает видео для ленивого чтения."""
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

        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            "LazyFrameExtractor: %dx%d, FPS=%.1f, интервал=%d кадров, "
            "ожидается %d кадров (ленивый режим, память минимальна)",
            self._width, self._height, self._fps, self._interval_frames, self.total_frames
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Закрывает VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self):
        """Итератор по кадрам видео."""
        return self

    def __next__(self) -> tuple[np.ndarray, float]:
        """Извлекает следующий кадр с заданным интервалом.

        Returns:
            Кортеж (frame_bgr, timestamp_ms).

        Raises:
            StopIteration: Когда кадры закончились или достигнут лимит.
        """
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

                # Проверяем лимит
                if self._max_frames is not None and self._extracted_count > self._max_frames:
                    raise StopIteration

                return frame, timestamp_ms


def extract_frames_lazy(
    video_path: str | Path,
    interval_ms: int = 500,
    max_frames: int | None = None,
) -> LazyFrameExtractor:
    """Создаёт ленивый извлекатель кадров (загрузка по требованию).

    В отличие от extract_frames(), НЕ загружает все кадры в память.
    Кадры извлекаются по одному при итерации.

    Args:
        video_path: Путь к видеофайлу.
        interval_ms: Интервал между кадрами в миллисекундах.
        max_frames: Максимальное количество кадров (None = без ограничений).

    Returns:
        LazyFrameExtractor — использовать с контекстным менеджером.

    Example:
        with extract_frames_lazy(video_path) as frames:
            for frame, ts_ms in frames:
                process(frame)
    """
    return LazyFrameExtractor(video_path, interval_ms, max_frames)


def get_video_info(video_path: str | Path) -> dict:
    """Возвращает базовую информацию о видеофайле.

    Args:
        video_path: Путь к видеофайлу.

    Returns:
        Словарь с метаданными видео.
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
        "duration_s": duration_s,
        "total_frames": total_frames,
    }


def _compute_frame_hash(frame: np.ndarray) -> str:
    """Вычисляет хэш кадра для кэширования.

    Использует первые 100 пикселей для быстрого хэширования.

    Args:
        frame: Кадр BGR.

    Returns:
        Строка-хэш кадра.
    """
    # Используем resize до маленького размера для стабильного хэша
    small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
    return str(hash(small.tobytes()))


def detect_video_type(
    frame: np.ndarray,
    prev_frame: np.ndarray | None = None,
    video_path: str | None = None,
) -> VideoType:
    """Определяет тип видеозаписи: direct / handheld / handheld_angle.

    Алгоритм учитывает яркость границ, разрешение, перспективу и стабильность.
    Результаты кэшируются по хэшу кадра для избежания повторных вычислений.

    Args:
        frame: Текущий кадр.
        prev_frame: Предыдущий кадр (для анализа стабильности).
        video_path: Путь к видео (опционально, для более точного кэширования).

    Returns:
        Тип видео (VideoType).
    """
    # Проверяем кэш
    cache_key = video_path if video_path else _compute_frame_hash(frame)
    if cache_key in _video_type_cache:
        return _video_type_cache[cache_key]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # 1. Анализ яркости границ
    top = gray[0:5, :].mean()
    bottom = gray[-5:, :].mean()
    left = gray[:, 0:5].mean()
    right = gray[:, -5:].mean()
    border_avg = (top + bottom + left + right) / 4.0

    # 2. Проверка разрешения
    width = w

    # 3. Детекция перспективы (горизонтальность текстовых линий)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10
    )
    angle_deviation = 0.0
    if lines is not None and len(lines) > 5:
        angles = [
            np.degrees(
                np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0])
            )
            for l in lines
        ]
        angle_deviation = float(np.std(angles))

    # 4. Анализ стабильности
    shake = 0.0
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        # Приводим к одному размеру
        if prev_gray.shape != gray.shape:
            prev_gray = cv2.resize(prev_gray, (w, h))
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        shake = float(np.mean(np.abs(flow)))

    # Классификация
    if border_avg > 150 and width > 1800 and shake < 0.5:
        result = VideoType.DIRECT
    elif angle_deviation > 2.0 or shake > 2.0:
        result = VideoType.HANDHELD_ANGLE
    else:
        result = VideoType.HANDHELD

    # Сохраняем в кэш
    _video_type_cache[cache_key] = result
    return result


def _ms_to_timestamp(ms: float) -> str:
    """Преобразует миллисекунды в формат HH:MM:SS.mmm.

    Args:
        ms: Время в миллисекундах.

    Returns:
        Строка таймстемпа в формате HH:MM:SS.mmm.
    """
    total_seconds = int(ms // 1000)
    millis = int(ms % 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def generate_video_id() -> str:
    """Генерирует уникальный ID для видео."""
    return str(uuid.uuid4())
