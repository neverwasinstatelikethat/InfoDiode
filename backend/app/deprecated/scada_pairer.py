"""SCADA proximity pairer для сопоставления меток и цветных индикаторов.

Алгоритм основан на пространственной близости с учётом направленных штрафов:
- Индикаторы значений имеют цветовые метки (value_green, value_blue, value_red, value_white)
- Метки параметров определяются по тексту (кириллица, ID сенсоров)
- KDTree для быстрого поиска ближайших соседей
- Жадное назначение с учётом направленных штрафов
"""

from __future__ import annotations

import re

import numpy as np
from scipy.spatial import KDTree

from app.core.ocr_models import ColorTag, TextBox, TextPair


class ScadaPairer:
    """Паттерн сопоставления для SCADA мнемосхем.

    Использует цветовые метки индикаторов для точного сопоставления
    с метками параметров на основе пространственной близости.
    """

    # Штрафы для направленных связей
    LEFT_PENALTY = 1000.0  # Штраф если значение слева от метки
    VERTICAL_PENALTY_BASE = 500.0  # Базовый штраф за вертикальное смещение

    # Максимальное количество кандидатов для поиска
    K_NEAREST = 5

    # Максимальное расстояние как доля от ширины кадра
    MAX_DISTANCE_RATIO = 0.20

    def pair(
        self,
        boxes: list[TextBox],
        frame_width: int,
        frame_height: int,
    ) -> list[TextPair]:
        """Сопоставляет метки с индикаторами значений по пространственной близости.

        Args:
            boxes: Список текстовых блоков с цветовыми метками.
            frame_width: Ширина кадра в пикселях.
            frame_height: Высота кадра в пикселях.

        Returns:
            Список пар метка-значение с уверенностью сопоставления.
        """
        if not boxes or frame_width <= 0 or frame_height <= 0:
            return []

        # Разделяем боксы на индикаторы и метки
        indicators = self._extract_indicators(boxes)
        labels = self._extract_labels(boxes)

        # Если нет индикаторов — возвращаем пустой список (fallback к другим методам)
        if not indicators:
            return []

        # Если нет меток — тоже возвращаем пустой список
        if not labels:
            return []

        # Строим KDTree по центрам индикаторов
        indicator_centers = np.array([box.bbox.center for box in indicators])
        tree = KDTree(indicator_centers)

        # Максимальное расстояние для кандидатов
        max_distance = frame_width * self.MAX_DISTANCE_RATIO

        # Масштабный коэффициент для вертикального штрафа
        height_scale = frame_height / 800.0

        # Жадное сопоставление
        used_indicators: set[int] = set()
        pairs: list[TextPair] = []

        for label_box in labels:
            label_center = label_box.bbox.center

            # Ищем K ближайших индикаторов
            k = min(self.K_NEAREST, len(indicators))
            distances, indices = tree.query(label_center, k=k)

            # Нормализуем к массивам если k=1
            if k == 1:
                distances = np.array([distances])
                indices = np.array([indices])

            best_candidate: tuple[float, TextBox, float] | None = None
            best_idx: int = -1

            for dist, idx in zip(distances, indices):
                if idx in used_indicators:
                    continue

                if dist > max_distance:
                    continue

                indicator_box = indicators[idx]
                indicator_center = indicator_box.bbox.center

                # Вычисляем направленные штрафы
                dx = indicator_center[0] - label_center[0]
                dy = indicator_center[1] - label_center[1]

                penalty = 0.0

                # Штраф если значение слева от метки (dx < -10 пикселей)
                if dx < -10:
                    penalty += self.LEFT_PENALTY

                # Штраф за большое вертикальное смещение
                vertical_threshold = 50.0 * height_scale
                if abs(dy) > vertical_threshold:
                    penalty += self.VERTICAL_PENALTY_BASE

                # Итоговый счёт = расстояние + штрафы
                score = dist + penalty

                if best_candidate is None or score < best_candidate[0]:
                    # Уверенность сопоставления уменьшается с ростом расстояния
                    confidence = max(0.0, 1.0 - (score / max_distance))
                    best_candidate = (score, indicator_box, confidence)
                    best_idx = idx

            # Если нашли подходящего кандидата — создаём пару
            if best_candidate is not None and best_idx >= 0:
                score, indicator_box, confidence = best_candidate
                used_indicators.add(best_idx)

                pairs.append(
                    TextPair(
                        label=label_box,
                        value=indicator_box,
                        relation="horizontal",
                        pair_confidence=confidence,
                    )
                )

        return pairs

    def _extract_indicators(self, boxes: list[TextBox]) -> list[TextBox]:
        """Извлекает боксы с цветовыми метками индикаторов значений.

        Args:
            boxes: Все текстовые блоки.

        Returns:
            Список боксов с color_tag начинающимся на "value_".
        """
        value_tags: set[ColorTag] = {
            "value_green",
            "value_blue",
            "value_red",
            "value_white",
            "value_yellow",
        }
        return [box for box in boxes if box.color_tag in value_tags]

    def _extract_labels(self, boxes: list[TextBox]) -> list[TextBox]:
        """Извлекает боксы с метками параметров.

        Метка определяется по:
        - Явной color_tag == "label"
        - Тексту с кириллицей (русские названия)
        - Тексту похожему на ID сенсора (TI-101, P-205)

        Args:
            boxes: Все текстовые блоки.

        Returns:
            Список боксов с метками параметров.
        """
        labels: list[TextBox] = []

        for box in boxes:
            # Явная метка по цвету
            if box.color_tag == "label":
                labels.append(box)
                continue

            # Пропускаем индикаторы значений
            if box.color_tag.startswith("value_"):
                continue

            # Проверяем текст на похожесть к метке
            text = box.text.strip()

            # Кириллица — русское название параметра
            if re.search(r"[а-яА-ЯёЁ]", text):
                labels.append(box)
                continue

            # ID сенсора: TI-101, P-205, dP-310, и т.д.
            if re.match(r"^[A-Za-z]{1,3}[-]?\d{2,5}$", text):
                labels.append(box)
                continue

            # OCR-ошибка: цифры + дефис (11-101 вместо TI-101)
            if re.match(r"^\d{1,2}-\d{2,5}$", text):
                labels.append(box)
                continue

        return labels
