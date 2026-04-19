"""Детектор экрана монитора на кадре видеозаписи.

Находит контуры монитора для handheld-видео и вычисляет
гомографию для коррекции перспективы.

Также включает функции для детекции активной вкладки SCADA-интерфейса.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Константы для детекции вкладок SCADA
TAB_REGION_HEIGHT_RATIO: Final[float] = 0.12  # Верхние 12% кадра для вкладок
TAB_BRIGHTNESS_THRESHOLD: Final[float] = 30.0  # Минимальная разница яркости
TAB_MIN_ACTIVE_RATIO: Final[float] = 1.15  # Минимальное отношение яркости активной вкладки
TAB_UNCERTAINTY_THRESHOLD: Final[float] = 1.05  # Порог неопределенности


def detect_screen(frame: np.ndarray) -> np.ndarray | None:
    """Обнаруживает экран монитора на кадре и возвращает 4 угловые точки.

    Используется для handheld-видео, где экран монитора виден
    через тёмную рамку (bezel).

    Args:
        frame: Входной кадр BGR.

    Returns:
        Массив 4 угловых точек (4,2) или None, если экран не найден.
    """
    t0 = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Блюрим для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Ищем тёмные границы (bezel монитора)
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

    # Морфологические операции для объединения областей
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Ищем контуры
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.debug("Экран не обнаружен: контуры не найдены")
        return None

    # Выбираем наибольший контур
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Контур должен занимать значительную часть кадра
    if area < (h * w * 0.3):
        logger.debug("Экран не обнаружен: площадь контура %.1f%% < 30%%", area / (h * w) * 100)
        return None

    # Аппроксимируем контур до 4 точек (четырёхугольник)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if len(approx) == 4:
        logger.debug("Экран обнаружен: 4 угла, площадь %.1f%%, %.1f мс", area / (h * w) * 100, elapsed_ms)
        return approx.reshape(4, 2).astype(np.float32)

    # Если не 4 точки, используем minAreaRect
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    logger.debug("Экран обнаружен (minAreaRect): площадь %.1f%%, %.1f мс", area / (h * w) * 100, elapsed_ms)
    return box.astype(np.float32)


def compute_homography(
    src_points: np.ndarray,
    dst_size: tuple[int, int],
) -> np.ndarray:
    """Вычисляет матрицу гомографии для коррекции перспективы.

    Args:
        src_points: 4 исходные точки (4,2).
        dst_size: Целевой размер (width, height).

    Returns:
        Матрица гомографии 3x3.
    """
    w, h = dst_size
    dst_points = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )

    # Упорядочиваем точки: верх-лево, верх-право, низ-право, низ-лево
    src_ordered = _order_points(src_points)

    return cv2.getPerspectiveTransform(src_ordered, dst_points)


def apply_homography(
    frame: np.ndarray,
    homography: np.ndarray,
    dst_size: tuple[int, int],
) -> np.ndarray:
    """Применяет гомографию к кадру.

    Args:
        frame: Входной кадр BGR.
        homography: Матрица гомографии 3x3.
        dst_size: Целевой размер (width, height).

    Returns:
        Кадр с исправленной перспективой.
    """
    return cv2.warpPerspective(frame, homography, dst_size)


def stabilize_frame(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
) -> np.ndarray:
    """Стабилизирует текущий кадр относительно предыдущего.

    Использует Lucas-Kanade оптический поток.

    Args:
        prev_frame: Предыдущий кадр.
        curr_frame: Текущий кадр.

    Returns:
        Стабилизированный кадр.
    """
    t0 = time.perf_counter()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    h, w = curr_gray.shape[:2]

    # Находим характерные точки
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)

    if prev_pts is None or len(prev_pts) < 10:
        logger.debug("Стабилизация пропущена: недостаточно точек (%s)", len(prev_pts) if prev_pts else 0)
        return curr_frame

    # Вычисляем оптический поток
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

    if curr_pts is None:
        logger.debug("Стабилизация пропущена: оптический поток не вычислен")
        return curr_frame

    # Фильтруем хорошие точки
    good_prev = prev_pts[status.flatten() == 1]
    good_curr = curr_pts[status.flatten() == 1]

    if len(good_prev) < 10:
        logger.debug("Стабилизация пропущена: недостаточно хороших точек (%d)", len(good_prev))
        return curr_frame

    # Вычисляем аффинное преобразование
    transform, inliers = cv2.estimateAffine2D(good_curr, good_prev)

    if transform is None:
        logger.debug("Стабилизация пропущена: не удалось оценить преобразование")
        return curr_frame

    # Применяем обратное преобразование для стабилизации
    stabilized = cv2.warpAffine(curr_frame, transform, (w, h))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.debug("Стабилизация выполнена: %d точек, %.1f мс", len(good_prev), elapsed_ms)

    return stabilized


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Упорядочивает 4 точки: верх-лево, верх-право, низ-право, низ-лево.

    Args:
        pts: 4 точки (4,2).

    Returns:
        Упорядоченные точки (4,2).
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


@dataclass
class TabDetectionResult:
    """Результат детекции активной вкладки SCADA.

    Attributes:
        tab_index: Индекс активной вкладки (0-3) или -1 если не определено
        confidence: Уверенность в результате (0.0-1.0)
        brightness_values: Список средней яркости для каждой вкладки
        method: Использованный метод детекции
    """

    tab_index: int
    confidence: float
    brightness_values: list[float]
    method: str


def detect_active_scada_tab(
    frame: np.ndarray,
    num_tabs: int = 4,
    region_height_ratio: float = TAB_REGION_HEIGHT_RATIO,
) -> TabDetectionResult:
    """Определяет активную вкладку SCADA-интерфейса на кадре.

    Анализирует верхнюю часть кадра (где обычно располагаются вкладки в SCADA)
    и определяет какая вкладка активна по яркости/цвету фона.
    Активная вкладка обычно имеет более яркий/светлый фон.

    Args:
        frame: Входной кадр BGR.
        num_tabs: Количество вкладок (по умолчанию 4 для 1_ai, 2_ai, 3_ai, 4_ai).
        region_height_ratio: Доля высоты кадра для анализа вкладок.

    Returns:
        TabDetectionResult с индексом активной вкладки и метаданными.
        tab_index = -1 если не удалось определить активную вкладку.
    """
    t0 = time.perf_counter()
    h, w = frame.shape[:2]

    # Определяем регион вкладок (верхняя часть кадра)
    tab_region_height = int(h * region_height_ratio)
    tab_region = frame[0:tab_region_height, 0:w]

    # Конвертируем в grayscale для анализа яркости
    gray_region = cv2.cvtColor(tab_region, cv2.COLOR_BGR2GRAY)

    # Разбиваем на сегменты по количеству вкладок
    tab_width = w // num_tabs
    brightness_values: list[float] = []

    for i in range(num_tabs):
        x_start = i * tab_width
        x_end = (i + 1) * tab_width if i < num_tabs - 1 else w

        tab_segment = gray_region[:, x_start:x_end]

        # Вычисляем среднюю яркость сегмента
        mean_brightness = float(np.mean(tab_segment))
        brightness_values.append(mean_brightness)

    # Находим вкладку с максимальной яркостью (предположительно активная)
    max_brightness = max(brightness_values)
    min_brightness = min(brightness_values)
    max_idx = brightness_values.index(max_brightness)

    # Анализируем разброс яркости для определения уверенности
    brightness_range = max_brightness - min_brightness
    brightness_std = np.std(brightness_values)

    # Определяем метод и уверенность
    method = "brightness"
    confidence = 0.0
    tab_index = -1

    # Проверяем, достаточно ли различима активная вкладка
    if brightness_range < TAB_BRIGHTNESS_THRESHOLD:
        # Разница слишком мала - возможно все вкладки одинаковые
        logger.debug(
            "Детекция вкладки: разница яркости %.1f < %.1f - неопределенность",
            brightness_range,
            TAB_BRIGHTNESS_THRESHOLD,
        )
        method = "uncertain_low_contrast"
    else:
        # Проверяем отношение яркости активной вкладки к средней остальных
        other_brightness = [b for i, b in enumerate(brightness_values) if i != max_idx]
        avg_other = sum(other_brightness) / len(other_brightness) if other_brightness else max_brightness

        if avg_other > 0:
            brightness_ratio = max_brightness / avg_other
        else:
            brightness_ratio = float("inf")

        if brightness_ratio < TAB_MIN_ACTIVE_RATIO:
            # Разница недостаточна для уверенности
            logger.debug(
                "Детекция вкладки: отношение яркости %.2f < %.2f - неопределенность",
                brightness_ratio,
                TAB_MIN_ACTIVE_RATIO,
            )
            method = "uncertain_ratio"
        elif brightness_ratio < TAB_UNCERTAINTY_THRESHOLD:
            # Граничный случай - умеренная уверенность
            tab_index = max_idx
            confidence = min(0.7, brightness_ratio - 1.0)
            method = "brightness_moderate"
        else:
            # Хорошее различие - высокая уверенность
            tab_index = max_idx
            confidence = min(1.0, 0.7 + (brightness_ratio - TAB_UNCERTAINTY_THRESHOLD) * 0.3)
            method = "brightness_high"

    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Детекция SCADA-вкладки: вкладка=%d, уверенность=%.2f, метод=%s, "
        "яркость=%s, время=%.1f мс",
        tab_index,
        confidence,
        method,
        [f"{b:.1f}" for b in brightness_values],
        elapsed_ms,
    )

    return TabDetectionResult(
        tab_index=tab_index,
        confidence=confidence,
        brightness_values=brightness_values,
        method=method,
    )


def detect_active_scada_tab_simple(
    frame: np.ndarray,
    num_tabs: int = 4,
) -> int:
    """Упрощенная версия детекции активной вкладки.

    Возвращает только индекс вкладки или -1 при неопределенности.
    Для использования в pipeline где нужен простой int результат.

    Args:
        frame: Входной кадр BGR.
        num_tabs: Количество вкладок.

    Returns:
        Индекс активной вкладки (0-3) или -1 если не определено.
    """
    result = detect_active_scada_tab(frame, num_tabs)
    return result.tab_index


def has_tab_changed(prev_tab: int, current_tab: int) -> bool:
    """Проверяет, изменилась ли активная вкладка.

    Args:
        prev_tab: Предыдущий индекс вкладки.
        current_tab: Текущий индекс вкладки.

    Returns:
        True если вкладка изменилась и текущая валидна (не -1).
    """
    if current_tab < 0:
        return False
    return prev_tab != current_tab


class ScadaTabTracker:
    """Трекер для отслеживания изменений вкладок SCADA на основе MSE и цветового анализа.

    Используется для детекции смены вкладок через сравнение заголовочной области
    и определения активной вкладки по цветовым признакам (HSV).
    """

    # Константы для анализа вкладок
    HEADER_HEIGHT_RATIO: float = 0.07  # Верхние 7% кадра для заголовка
    MSE_THRESHOLD: float = 1000.0  # Порог MSE для определения смены вкладки
    HEADER_RESIZE_WIDTH: int = 320  # Ширина для ресайза заголовка (скорость)
    HEADER_RESIZE_HEIGHT: int = 24  # Высота для ресайза заголовка

    # HSV диапазоны для детекции активной вкладки (синий/голубой фон)
    BLUE_HSV_LOWER: np.ndarray = np.array([90, 50, 50])  # Нижняя граница синего
    BLUE_HSV_UPPER: np.ndarray = np.array([130, 255, 255])  # Верхняя граница синего
    CYAN_HSV_LOWER: np.ndarray = np.array([80, 50, 50])  # Нижняя граница голубого
    CYAN_HSV_UPPER: np.ndarray = np.array([100, 255, 255])  # Верхняя граница голубого

    def __init__(self) -> None:
        """Инициализирует трекер вкладок.

        Сохраняет предыдущий заголовочный регион как numpy array.
        """
        self._prev_header: np.ndarray | None = None

    def check_tab_change(self, frame: np.ndarray) -> bool:
        """Проверяет, изменилась ли активная вкладка на основе MSE заголовка.

        Извлекает верхние 6-8% кадра как заголовочный регион, вычисляет MSE
        с предыдущим заголовком. Если MSE > порога — вкладка изменилась.

        Args:
            frame: Входной кадр BGR.

        Returns:
            True если вкладка изменилась, иначе False.
        """
        h, w = frame.shape[:2]

        # Извлекаем заголовочный регион (верхние 6-8% кадра)
        header_height = int(h * self.HEADER_HEIGHT_RATIO)
        header_region = frame[0:header_height, 0:w]

        # Конвертируем в grayscale для сравнения
        header_gray = cv2.cvtColor(header_region, cv2.COLOR_BGR2GRAY)

        # Ресайзим до фиксированного размера для скорости
        header_resized = cv2.resize(
            header_gray,
            (self.HEADER_RESIZE_WIDTH, self.HEADER_RESIZE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        # Если нет предыдущего заголовка — сохраняем текущий и возвращаем False
        if self._prev_header is None:
            self._prev_header = header_resized.copy()
            logger.debug("Трекер вкладок: инициализация первого заголовка")
            return False

        # Вычисляем MSE между текущим и предыдущим заголовком
        mse = np.mean((header_resized.astype(np.float32) - self._prev_header.astype(np.float32)) ** 2)

        # Проверяем порог MSE
        if mse > self.MSE_THRESHOLD:
            logger.info(
                "Трекер вкладок: обнаружена смена вкладки (MSE=%.1f > %.1f)",
                mse,
                self.MSE_THRESHOLD,
            )
            self._prev_header = header_resized.copy()
            return True

        logger.debug("Трекер вкладок: вкладка не изменилась (MSE=%.1f)", mse)
        return False

    def detect_active_tab(self, frame: np.ndarray) -> str | None:
        """Определяет активную вкладку SCADA по цветовым признакам.

        Анализирует заголовочную полосу, находит подсвеченную (синий/голубой
        фон) область через HSV-анализ и возвращает имя активной вкладки.

        Args:
            frame: Входной кадр BGR.

        Returns:
            Имя активной вкладки как строка, или None если не определено.
        """
        h, w = frame.shape[:2]

        # Извлекаем заголовочную полосу (верхние 6-8%)
        header_height = int(h * self.HEADER_HEIGHT_RATIO)
        header_region = frame[0:header_height, 0:w]

        # Конвертируем в HSV для цветового анализа
        hsv_region = cv2.cvtColor(header_region, cv2.COLOR_BGR2HSV)

        # Создаём маску для синего/голубого цвета (активная вкладка)
        blue_mask = cv2.inRange(hsv_region, self.BLUE_HSV_LOWER, self.BLUE_HSV_UPPER)
        cyan_mask = cv2.inRange(hsv_region, self.CYAN_HSV_LOWER, self.CYAN_HSV_UPPER)
        active_mask = cv2.bitwise_or(blue_mask, cyan_mask)

        # Морфологические операции для очистки маски
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        active_mask = cv2.morphologyEx(active_mask, cv2.MORPH_CLOSE, kernel)
        active_mask = cv2.morphologyEx(active_mask, cv2.MORPH_OPEN, kernel)

        # Ищем контуры выделенной области
        contours, _ = cv2.findContours(
            active_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.debug("Детекция вкладки: не найдены контуры активной вкладки")
            return None

        # Выбираем наибольший контур (предположительно активная вкладка)
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)

        # Проверяем минимальную площадь контура
        min_area = (w * header_height) * 0.05  # Минимум 5% от заголовка
        if contour_area < min_area:
            logger.debug(
                "Детекция вкладки: контур слишком мал (%.1f < %.1f)",
                contour_area,
                min_area,
            )
            return None

        # Получаем bounding box выделенной области
        x, y, tab_w, tab_h = cv2.boundingRect(largest_contour)

        # Извлекаем ROI с текстом вкладки
        tab_roi = header_region[y : y + tab_h, x : x + tab_w]

        # Конвертируем в grayscale для OCR
        tab_gray = cv2.cvtColor(tab_roi, cv2.COLOR_BGR2GRAY)

        # Применяем бинаризацию для улучшения OCR
        _, tab_binary = cv2.threshold(tab_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Пытаемся распознать текст с помощью PaddleOCR (если доступен)
        # Или используем простую эвристику для определения имени вкладки
        tab_name = self._extract_tab_name_from_position(x, w)

        logger.info(
            "Детекция активной вкладки: найдена вкладка '%s' (позиция=%d, площадь=%.1f)",
            tab_name,
            x,
            contour_area,
        )

        return tab_name

    def _extract_tab_name_from_position(self, x_position: int, frame_width: int) -> str:
        """Извлекает имя вкладки на основе позиции в заголовке.

        SCADA-интерфейс обычно имеет 4 вкладки: 1_ai, 2_ai, 3_ai, 4_ai.
        Определяет вкладку по горизонтальной позиции.

        Args:
            x_position: Горизонтальная позиция левого края вкладки.
            frame_width: Ширина кадра.

        Returns:
            Имя вкладки (1_ai, 2_ai, 3_ai, 4_ai или unknown).
        """
        # Разбиваем заголовок на 4 равные части
        tab_width = frame_width / 4

        if x_position < tab_width:
            return "1_ai"
        elif x_position < 2 * tab_width:
            return "2_ai"
        elif x_position < 3 * tab_width:
            return "3_ai"
        elif x_position < 4 * tab_width:
            return "4_ai"
        else:
            return "unknown"

    def reset(self) -> None:
        """Сбрасывает трекер, очищая сохранённый заголовок."""
        self._prev_header = None
        logger.debug("Трекер вкладок: сброшен")
