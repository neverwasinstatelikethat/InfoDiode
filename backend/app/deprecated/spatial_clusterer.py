"""Пространственный кластеризатор для связывания меток и значений.

Параметры на мнемосхеме SCADA расположены хаотично — НЕ в таблицах.
Каждый параметр представлен парой: метка (TI-101) и значение (758.3),
расположенные рядом друг с другом (50-100px по вертикали).

Алгоритм (v2):
1. Конвертация OCRTextResult → TextBox (новый контракт данных)
2. TextClassifier: классификация label/value/mixed
3. ProximityGraph: KDTree-граф пространственных связей
4. PairExtractor: жадное назначение + табличные строки
5. Конвертация TextPair → LabelValuePair (обратная совместимость)

Также поддерживает парсинг правой панели (2-колоночный layout).
"""

from __future__ import annotations

import logging
import re

import numpy as np

from app.models.schemas import BoundingBox, LabelValuePair, OCRTextResult, ZoneType


def cluster_label_value_pairs(
    ocr_results: list[OCRTextResult],
    frame_shape: tuple[int, int],
    max_y_distance_px: int | None = None,
    min_x_overlap_ratio: float = 0.15,
) -> list[LabelValuePair]:
    """Связывает распознанные тексты в пары метка-значение.

    Использует улучшенный алгоритм: TextClassifier → ProximityGraph → PairExtractor.
    Fallback на простой алгоритм при ошибке.

    Args:
        ocr_results: Результаты OCR на кадре.
        frame_shape: Размер кадра (height, width).
        max_y_distance_px: Максимальное расстояние по Y (для fallback).
            Если None, вычисляется как 8% от высоты кадра.
        min_x_overlap_ratio: Минимальное перекрытие по X (для fallback).
            Уменьшено до 0.15 для scattered SCADA параметров.

    Returns:
        Список пар метка-значение.
    """
    # Вычисляем адаптивное расстояние по Y (8% от высоты кадра)
    h, _ = frame_shape[:2]
    if max_y_distance_px is None:
        max_y_distance_px = int(h * 0.08)
    if not ocr_results:
        return []

    # Пробуем улучшенный алгоритм
    try:
        return _cluster_enhanced(ocr_results, frame_shape)
    except Exception as e:
        logger.warning("Улучшенный кластеризатор упал, fallback: %s", e)
        return _cluster_simple(ocr_results, frame_shape, max_y_distance_px, min_x_overlap_ratio)


def parse_right_panel(
    ocr_results: list[OCRTextResult],
    zone_bbox: BoundingBox,
    frame_shape: tuple[int, int],
) -> list[LabelValuePair]:
    """Парсит правую панель как структурированные label-value пары.

    Правая панель содержит 2-колоночный layout:
    слева — название параметра, справа — значение + единица.

    Args:
        ocr_results: Результаты OCR.
        zone_bbox: Границы правой панели.
        frame_shape: Размер кадра.

    Returns:
        Список пар метка-значение из правой панели.
    """
    h, w = frame_shape[:2]

    # Фильтруем результаты внутри зоны
    zone_results = [
        r
        for r in ocr_results
        if _is_inside_zone(r.bbox, zone_bbox)
    ]

    if len(zone_results) < 2:
        return []

    # Разделяем зону на левую (метки) и правую (значения) половины
    mid_x = (zone_bbox.x1 + zone_bbox.x2) / 2

    left_items: list[OCRTextResult] = []
    right_items: list[OCRTextResult] = []

    for result in zone_results:
        cx = (result.bbox.x1 + result.bbox.x2) / 2
        if cx < mid_x:
            left_items.append(result)
        else:
            right_items.append(result)

    # Связываем по Y-координате
    pairs: list[LabelValuePair] = []
    used_right: set[int] = set()

    for left in left_items:
        left_cy = (left.bbox.y1 + left.bbox.y2) / 2
        best_idx = -1
        best_dist = float("inf")

        for ridx, right in enumerate(right_items):
            if ridx in used_right:
                continue
            right_cy = (right.bbox.y1 + right.bbox.y2) / 2
            dist = abs(left_cy - right_cy)
            if dist < 0.02 and dist < best_dist:  # ~2% высоты кадра
                best_dist = dist
                best_idx = ridx

        if best_idx >= 0:
            used_right.add(best_idx)
            right = right_items[best_idx]

            pairs.append(
                LabelValuePair(
                    label=left.text,
                    value=_clean_value(right.text),
                    label_bbox=left.bbox,
                    value_bbox=right.bbox,
                    confidence=min(left.confidence, right.confidence),
                    zone=ZoneType.RIGHT_PANEL,
                    color_state="normal",
                )
            )

    return pairs


# ---------------------------------------------------------------------------
# Улучшенный алгоритм (TextClassifier + ProximityGraph + PairExtractor)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _cluster_enhanced(
    ocr_results: list[OCRTextResult],
    frame_shape: tuple[int, int],
) -> list[LabelValuePair]:
    """Улучшенный алгоритм кластеризации через новый OCR pipeline.

    Алгоритм:
    1. Сначала пробуем ScadaPairer (если есть цветовые метки индикаторов)
    2. Если ScadaPairer не дал результатов — fallback к PairExtractor
    """
    from app.core.ocr_models import BBox, TextBox
    from app.core.pair_extractor import PairExtractor
    from app.core.scada_pairer import ScadaPairer

    h, w = frame_shape[:2]

    # Конвертируем OCRTextResult → TextBox
    text_boxes: list[TextBox] = []
    for r in ocr_results:
        x1 = int(r.bbox.x1 * w)
        y1 = int(r.bbox.y1 * h)
        x2 = int(r.bbox.x2 * w)
        y2 = int(r.bbox.y2 * h)
        bbox = BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
        text_boxes.append(
            TextBox(
                bbox=bbox,
                text=r.text,
                confidence=r.confidence,
                source="paddle",
            )
        )

    # Сначала пробуем ScadaPairer (primary path для SCADA мнемосхем)
    scada_pairer = ScadaPairer()
    scada_pairs = scada_pairer.pair(text_boxes, w, h)

    # Если ScadaPairer вернул пары — используем их
    if scada_pairs:
        return _convert_text_pairs_to_label_value_pairs(scada_pairs, h, w)

    # Fallback: используем PairExtractor (column/proximity/table paths)
    extractor = PairExtractor()
    text_pairs = extractor.extract(text_boxes)

    return _convert_text_pairs_to_label_value_pairs(text_pairs, h, w)


def _convert_text_pairs_to_label_value_pairs(
    text_pairs: list[TextPair],
    h: int,
    w: int,
) -> list[LabelValuePair]:
    """Конвертирует TextPair в LabelValuePair с нормализацией координат.

    Args:
        text_pairs: Список пар из OCR pipeline.
        h: Высота кадра.
        w: Ширина кадра.

    Returns:
        Список LabelValuePair с нормализованными координатами.
    """
    from app.core.ocr_models import TextPair

    pairs: list[LabelValuePair] = []
    for tp in text_pairs:
        # Пиксельные → нормализованные координаты
        lb = tp.label.bbox
        vb = tp.value.bbox
        pairs.append(
            LabelValuePair(
                label=tp.label.text,
                value=_clean_value(tp.value.text),
                label_bbox=BoundingBox(
                    x1=lb.x / max(w, 1), y1=lb.y / max(h, 1),
                    x2=(lb.x + lb.w) / max(w, 1), y2=(lb.y + lb.h) / max(h, 1),
                ),
                value_bbox=BoundingBox(
                    x1=vb.x / max(w, 1), y1=vb.y / max(h, 1),
                    x2=(vb.x + vb.w) / max(w, 1), y2=(vb.y + vb.h) / max(h, 1),
                ),
                confidence=tp.pair_confidence,
                zone=ZoneType.CENTRAL_SCHEMA,
                color_state="normal",
            )
        )

    return pairs


# ---------------------------------------------------------------------------
# Простой алгоритм (fallback)
# ---------------------------------------------------------------------------


def _cluster_simple(
    ocr_results: list[OCRTextResult],
    frame_shape: tuple[int, int],
    max_y_distance_px: int = 100,
    min_x_overlap_ratio: float = 0.15,
) -> list[LabelValuePair]:
    """Простой алгоритм: Y-distance + X-overlap + horizontal pairing (оригинальный + улучшения)."""
    h, w = frame_shape[:2]

    labels: list[tuple[int, OCRTextResult]] = []
    values: list[tuple[int, OCRTextResult]] = []

    for idx, result in enumerate(ocr_results):
        if _is_label_like(result.text):
            labels.append((idx, result))
        elif _is_value_like(result.text):
            values.append((idx, result))

    pairs: list[LabelValuePair] = []
    used_values: set[int] = set()

    for label_idx, label in labels:
        best_value_pos = -1
        best_distance = float("inf")
        best_mode = "vertical"  # "vertical" или "horizontal"

        label_cx = (label.bbox.x1 + label.bbox.x2) / 2
        label_cy = (label.bbox.y1 + label.bbox.y2) / 2
        label_width = (label.bbox.x2 - label.bbox.x1) * w

        for pos, (value_idx, value) in enumerate(values):
            if pos in used_values:
                continue

            value_cx = (value.bbox.x1 + value.bbox.x2) / 2
            value_cy = (value.bbox.y1 + value.bbox.y2) / 2

            # Режим 1: Вертикальное сопоставление (значение ниже метки)
            if value_cy > label_cy:
                y_dist_px = (value_cy - label_cy) * h
                if y_dist_px <= max_y_distance_px:
                    overlap = _x_overlap_ratio(label.bbox, value.bbox)
                    if overlap >= min_x_overlap_ratio:
                        dist = np.sqrt((label_cx - value_cx) ** 2 + (label_cy - value_cy) ** 2)
                        if dist < best_distance:
                            best_distance = dist
                            best_value_pos = pos
                            best_mode = "vertical"

            # Режим 2: Горизонтальное сопоставление (значение справа от метки, на той же линии)
            # SCADA: "Label Value" на одной линии
            y_dist_px = abs(value_cy - label_cy) * h
            x_dist_px = (value_cx - label_cx) * w
            if y_dist_px < max_y_distance_px * 0.5 and x_dist_px > 0:
                # Значение должно быть справа от метки, в пределах разумного расстояния
                max_x_distance = max(label_width * 3, w * 0.15)
                if x_dist_px <= max_x_distance:
                    dist = np.sqrt((label_cx - value_cx) ** 2 + (label_cy - value_cy) ** 2)
                    if dist < best_distance:
                        best_distance = dist
                        best_value_pos = pos
                        best_mode = "horizontal"

            # Режим 3: Справа-налево сопоставление (для правой панели, значение слева от метки)
            if y_dist_px < max_y_distance_px * 0.5 and x_dist_px < 0:
                max_x_distance = max(label_width * 2, w * 0.1)
                if abs(x_dist_px) <= max_x_distance:
                    dist = np.sqrt((label_cx - value_cx) ** 2 + (label_cy - value_cy) ** 2)
                    if dist < best_distance:
                        best_distance = dist
                        best_value_pos = pos
                        best_mode = "right_to_left"

        if best_value_pos >= 0:
            used_values.add(best_value_pos)
            value_result = values[best_value_pos][1]

            pairs.append(
                LabelValuePair(
                    label=label.text,
                    value=_clean_value(value_result.text),
                    label_bbox=label.bbox,
                    value_bbox=value_result.bbox,
                    confidence=min(label.confidence, value_result.confidence),
                    zone=ZoneType.CENTRAL_SCHEMA,
                    color_state="normal",
                )
            )

    return pairs


def _is_label_like(text: str) -> bool:
    """Определяет, похож ли текст на метку параметра.

    Метки содержат буквы: TI-101, P-205, dP-310, и т.д.
    Также ловит OCR-ошибки: 11-101, T1-101.

    Args:
        text: Текст для проверки.

    Returns:
        True если текст похож на метку.
    """
    # Содержит латинские буквы (SENSOR ID pattern)
    if re.match(r"^[A-Za-z]{1,3}[-]?\d{2,5}$", text.strip()):
        return True
    # OCR-ошибка: цифры + дефис (11-101 вместо TI-101)
    if re.match(r"^\d{1,2}-\d{2,5}$", text.strip()):
        return True
    # Содержит кириллицу (русское название параметра)
    if re.search(r"[а-яА-Я]", text):
        return True
    # Содержит буквы с дефисом
    if re.match(r"^[A-Za-z]{1,4}-\d", text.strip()):
        return True
    return False


def _is_value_like(text: str) -> bool:
    """Определяет, похож ли текст на числовое значение.

    Values: 758.3, 45.2, 101, -12.5, 0.25

    Args:
        text: Текст для проверки.

    Returns:
        True если текст похож на число.
    """
    cleaned = text.strip().replace(",", ".")
    return bool(re.match(r"^-?\d+\.?\d*$", cleaned))


def _clean_value(text: str) -> str:
    """Очищает значение от лишних символов.

    Args:
        text: Сырой текст значения.

    Returns:
        Очищенное значение.
    """
    cleaned = text.strip().replace(",", ".")
    # Удаляем единицы измерения в конце
    cleaned = re.sub(r"\s*(°C|кПа|мм/с|%|Гц|В|ед\.?)$", "", cleaned, flags=re.IGNORECASE)
    match = re.search(r"-?\d+\.?\d*", cleaned)
    return match.group(0) if match else cleaned


def _x_overlap_ratio(a: BoundingBox, b: BoundingBox) -> float:
    """Вычисляет отношение перекрытия двух bounding box по X.

    Args:
        a: Первый bbox.
        b: Второй bbox.

    Returns:
        Отношение перекрытия [0, 1].
    """
    overlap_start = max(a.x1, b.x1)
    overlap_end = min(a.x2, b.x2)
    if overlap_end <= overlap_start:
        return 0.0
    overlap = overlap_end - overlap_start
    width_a = a.x2 - a.x1
    width_b = b.x2 - b.x1
    return overlap / max(width_a, width_b, 1e-6)


def _is_inside_zone(bbox: BoundingBox, zone: BoundingBox) -> bool:
    """Проверяет, находится ли bbox внутри зоны.

    Args:
        bbox: Bounding box элемента.
        zone: Границы зоны.

    Returns:
        True если центр bbox внутри зоны.
    """
    cx = (bbox.x1 + bbox.x2) / 2
    cy = (bbox.y1 + bbox.y2) / 2
    return zone.x1 <= cx <= zone.x2 and zone.y1 <= cy <= zone.y2
