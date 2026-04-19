"""Модуль для цветовой фильтрации и классификации текстовых блоков.

Обнаруживает цветные индикаторные области на SCADA-экранах и классифицирует
распознанный текст по цветовым признакам (зелёные, синие, белые, красные индикаторы).
"""

from __future__ import annotations

import re

import cv2
import numpy as np

from app.core.ocr_models import BBox, ColorTag, TextBox


class ColorFilter:
    """Фильтр для обнаружения цветных индикаторных областей и классификации текстовых блоков.

    Использует HSV-цветовое пространство для надежного обнаружения индикаторов
    различных цветов (зелёных, голубых, белых, красных) на SCADA-экранах.
    """

    # HSV диапазоны для различных цветов индикаторов
    # Формат: (H_min, H_max, S_min, S_max, V_min, V_max)
    _GREEN_RANGE = (35, 85, 50, 255, 80, 255)  # Зелёные индикаторы (GPA-11)
    _BLUE_RANGE = (85, 130, 20, 255, 150, 255)  # Голубые/циановые (GPA-21)
    _WHITE_RANGE = (0, 179, 0, 30, 200, 255)  # Белые индикаторы (высокая V, низкая S)
    _RED_RANGE_1 = (0, 10, 100, 255, 80, 255)  # Красные (первая часть спектра)
    _RED_RANGE_2 = (170, 180, 100, 255, 80, 255)  # Красные (вторая часть спектра)

    # Параметры фильтрации контуров
    _MIN_AREA = 200  # Минимальная площадь индикатора в пикселях
    _MIN_ASPECT_RATIO = 1.5  # Минимальное соотношение сторон (ширина/высота)
    _MAX_ASPECT_RATIO = 10.0  # Максимальное соотношение сторон

    def __init__(self) -> None:
        """Инициализирует ColorFilter с предкомпилированными масками."""
        self._cached_frame_shape: tuple[int, int] | None = None
        self._cached_indicator_mask: np.ndarray | None = None
        self._cached_color_masks: dict[str, np.ndarray] = {}

    def detect_indicator_regions(
        self,
        frame: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """Обнаруживает цветные индикаторные области на кадре.

        Преобразует кадр в HSV, применяет цветовые маски для поиска
        индикаторов различных цветов, выполняет морфологические операции
        и возвращает список ограничивающих прямоугольников.

        Args:
            frame: Исходный кадр в формате BGR (OpenCV по умолчанию).

        Returns:
            Список кортежей (x, y, w, h) для каждой найденной индикаторной области.
        """
        if frame.size == 0:
            return []

        # Преобразуем в HSV для устойчивого цветового поиска
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Создаём маски для каждого цвета
        masks: dict[str, np.ndarray] = {}

        # Зелёные индикаторы
        masks["green"] = self._create_hsv_mask(hsv, self._GREEN_RANGE)

        # Голубые/циановые индикаторы
        masks["blue"] = self._create_hsv_mask(hsv, self._BLUE_RANGE)

        # Белые индикаторы
        masks["white"] = self._create_hsv_mask(hsv, self._WHITE_RANGE)

        # Красные индикаторы (объединяем два диапазона)
        red_mask_1 = self._create_hsv_mask(hsv, self._RED_RANGE_1)
        red_mask_2 = self._create_hsv_mask(hsv, self._RED_RANGE_2)
        masks["red"] = cv2.bitwise_or(red_mask_1, red_mask_2)

        # Объединяем все маски для поиска всех индикаторов
        combined_mask = np.zeros_like(masks["green"])
        for mask in masks.values():
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Морфологические операции для закрытия пробелов в масках
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Кэшируем маски для последующего использования в classify_textboxes
        self._cached_frame_shape = (frame.shape[0], frame.shape[1])
        self._cached_indicator_mask = combined_mask
        self._cached_color_masks = masks

        # Находим контуры
        contours, _ = cv2.findContours(
            combined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions: list[tuple[int, int, int, int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue

            aspect_ratio = w / h
            if not (self._MIN_ASPECT_RATIO <= aspect_ratio <= self._MAX_ASPECT_RATIO):
                continue

            regions.append((x, y, w, h))

        return regions

    def classify_textboxes(
        self,
        frame: np.ndarray,
        boxes: list[TextBox],
    ) -> list[TextBox]:
        """Классифицирует текстовые блоки по цветовым признакам индикаторов.

        Для каждого TextBox определяет, попадает ли его центр в область
        цветного индикатора, и присваивает соответствующую цветовую метку.
        Числовые значения вне индикаторов помечаются как подозрительные.

        Args:
            frame: Исходный кадр в формате BGR.
            boxes: Список текстовых блоков для классификации.

        Returns:
            Тот же список TextBox с обновлёнными полями color_tag.
        """
        if not boxes or frame.size == 0:
            return boxes

        # Обновляем кэш масок, если кадр изменился
        current_shape = (frame.shape[0], frame.shape[1])
        if (
            self._cached_frame_shape != current_shape
            or self._cached_indicator_mask is None
        ):
            self.detect_indicator_regions(frame)

        # Если маски не созданы, возвращаем как есть
        if self._cached_color_masks is None:
            return boxes

        # Получаем индивидуальные маски для каждого цвета
        green_mask = self._cached_color_masks.get("green", np.zeros_like(frame[:, :, 0]))
        blue_mask = self._cached_color_masks.get("blue", np.zeros_like(frame[:, :, 0]))
        white_mask = self._cached_color_masks.get("white", np.zeros_like(frame[:, :, 0]))
        red_mask = self._cached_color_masks.get("red", np.zeros_like(frame[:, :, 0]))

        for box in boxes:
            center_x, center_y = box.bbox.center
            cx, cy = int(center_x), int(center_y)

            # Проверяем границы кадра
            if not (
                0 <= cy < frame.shape[0]
                and 0 <= cx < frame.shape[1]
            ):
                box.color_tag = "unknown"
                continue

            # Определяем цвет индикатора по маскам (в порядке приоритета)
            if green_mask[cy, cx] > 0:
                box.color_tag = "value_green"
            elif blue_mask[cy, cx] > 0:
                box.color_tag = "value_blue"
            elif white_mask[cy, cx] > 0:
                box.color_tag = "value_white"
            elif red_mask[cy, cx] > 0:
                box.color_tag = "value_red"
            elif self._is_label_like(box.text):
                # Текст похож на метку (не число) и не в индикаторе
                box.color_tag = "label"
            else:
                # Числовое значение вне индикатора - подозрительно
                if self._is_numeric_value(box.text):
                    box.color_tag = "unknown"
                else:
                    box.color_tag = "unknown"

        return boxes

    def _create_hsv_mask(
        self,
        hsv: np.ndarray,
        hsv_range: tuple[int, int, int, int, int, int],
    ) -> np.ndarray:
        """Создаёт бинарную маску для заданного HSV-диапазона.

        Args:
            hsv: Кадр в цветовом пространстве HSV.
            hsv_range: Кортеж (H_min, H_max, S_min, S_max, V_min, V_max).

        Returns:
            Бинарная маска (numpy array типа uint8).
        """
        h_min, h_max, s_min, s_max, v_min, v_max = hsv_range

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        return cv2.inRange(hsv, lower, upper)

    def _is_label_like(self, text: str) -> bool:
        """Проверяет, похож ли текст на метку/название параметра.

        Метки обычно содержат буквы, спецсимволы (°С, кПа и т.д.)
        и не являются чистыми числами.

        Args:
            text: Текст для анализа.

        Returns:
            True если текст похож на метку, False если на значение.
        """
        cleaned = text.strip()
        if not cleaned:
            return False

        # Если есть буквы (включая кириллицу) - это метка
        if re.search(r"[a-zA-Zа-яА-ЯёЁ]", cleaned):
            return True

        # Спецсимволы типичные для обозначений
        label_chars = set("°С©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿×÷‐‑‒–—―‖‗'")
        if any(c in label_chars for c in cleaned):
            return True

        return False

    def _is_numeric_value(self, text: str) -> bool:
        """Проверяет, является ли текст числовым значением.

        Args:
            text: Текст для анализа.

        Returns:
            True если текст выглядит как число (целое или с плавающей точкой).
        """
        cleaned = text.strip().replace(",", ".")
        if not cleaned:
            return False

        # Убираем возможные единицы измерения (оставляем только число)
        # Паттерн: число с опциональным знаком и десятичной точкой
        numeric_pattern = r"^[+-]?\d+(?:\.\d+)?$"
        return bool(re.match(numeric_pattern, cleaned))

    def get_color_mask(self, color: str) -> np.ndarray | None:
        """Возвращает кэшированную маску для указанного цвета.

        Args:
            color: Название цвета ('green', 'blue', 'white', 'red').

        Returns:
            Бинарная маска или None если маска не создана.
        """
        return self._cached_color_masks.get(color)

    def clear_cache(self) -> None:
        """Очищает кэшированные маски для освобождения памяти."""
        self._cached_frame_shape = None
        self._cached_indicator_mask = None
        self._cached_color_masks = {}


def classify_boxes_by_color(
    frame: np.ndarray,
    boxes: list[TextBox],
) -> list[TextBox]:
    """Утилитарная функция для быстрой классификации текстовых блоков.

    Args:
        frame: Исходный кадр в формате BGR.
        boxes: Список текстовых блоков для классификации.

    Returns:
            Список TextBox с заполненными полями color_tag.
    """
    filter_instance = ColorFilter()
    return filter_instance.classify_textboxes(frame, boxes)
