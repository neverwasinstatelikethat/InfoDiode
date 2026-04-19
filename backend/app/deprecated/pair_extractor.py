"""Извлекатель пар label:value из графа пространственных связей.

Алгоритм:
1. Разбивает mixed-боксы («Давление: 12.3») на label + value
2. Классифицирует TextBox-ы через TextClassifier
3. Строит граф связей через ProximityGraph
4. Жадное назначение: каждый value → ближайший label
5. Детекция табличных строк (Y-группировка)
6. Слияние результатов
"""

from __future__ import annotations

from collections import defaultdict

from app.core.column_layout_analyzer import ColumnLayoutAnalyzer, LayoutConfig
from app.core.ocr_models import TextBox, TextPair
from app.core.proximity_graph import ProximityGraph
from app.core.text_classifier import TextClassifier


class PairExtractor:
    """Основная логика: из графа связей строит финальные TextPair."""

    def __init__(self, layout_config: LayoutConfig | None = None) -> None:
        self._classifier = TextClassifier()
        self._graph = ProximityGraph()
        self._column_analyzer = ColumnLayoutAnalyzer(config=layout_config)

    def extract(self, boxes: list[TextBox]) -> list[TextPair]:
        """Извлекает пары label:value из списка TextBox.

        Args:
            boxes: Распознанные текстовые блоки.

        Returns:
            Список пар метка-значение.
        """
        if not boxes:
            return []

        # Step 1: classify ALL boxes first (so box_type is filled)
        pre_classified = self._classifier.classify_all(boxes)

        # Step 2: expand mixed boxes (NOW box_type is filled, so split_inline works)
        expanded = self._expand_mixed(pre_classified)

        # Step 3: re-classify ONLY the new boxes from expansion
        new_boxes = [b for b in expanded if b not in pre_classified]
        for b in new_boxes:
            self._classifier.classify(b, expanded)
        classified = expanded

        # Step 4: Column-aware layout (PRIMARY path, reuses pre-created analyzer)
        column_pairs = self._column_analyzer.extract_pairs(classified)

        # Step 5: ProximityGraph as supplement (catches what column analyzer missed)
        edges = self._graph.build_edges(classified)
        valid_edges = self._filter_edges(edges, classified)
        proximity_pairs = self._greedy_assign(valid_edges)

        # Step 6: Table rows
        table_pairs = self._detect_table_rows(classified)

        # Step 7: Merge — priority to column_pairs
        return self._merge_pairs_three(column_pairs, proximity_pairs, table_pairs)

    # -----------------------------------------------------------------------
    # Внутренние методы
    # -----------------------------------------------------------------------

    def _expand_mixed(self, boxes: list[TextBox]) -> list[TextBox]:
        """Разбивает mixed-боксы на label + value."""
        result: list[TextBox] = []
        for box in boxes:
            if box.box_type == "mixed":
                split = self._classifier.split_inline(box)
                if split:
                    result.extend(split)
                else:
                    result.append(box)
            else:
                result.append(box)
        return result

    def _filter_edges(
        self,
        edges: list[tuple],
        boxes: list[TextBox],
    ) -> list[tuple]:
        """Оставляем рёбра label→value, отбрасываем остальные."""
        box_type = {id(b): b.box_type for b in boxes}
        valid: list[tuple] = []
        for box_a, box_b, relation, weight in edges:
            t_a = box_type.get(id(box_a), "unknown")
            t_b = box_type.get(id(box_b), "unknown")
            # Допускаем: label→value, label→unknown, unknown→value
            if t_a in ("label", "unknown") and t_b in ("value", "unknown"):
                valid.append((box_a, box_b, relation, weight))
        return valid

    def _greedy_assign(self, edges: list[tuple]) -> list[TextPair]:
        """Жадное назначение: для каждого value берём ребро с max weight.

        Каждый label используется не более одного раза.
        Уверенность пары ограничена диапазоном [0, 1].
        """
        # value_id → список рёбер
        value_edges: dict[int, list] = defaultdict(list)
        for box_a, box_b, relation, weight in edges:
            value_edges[id(box_b)].append((weight, box_a, box_b, relation))

        used_labels: set[int] = set()
        pairs: list[TextPair] = []

        # Обрабатываем в порядке убывания веса лучшего ребра
        sorted_values = sorted(
            value_edges.items(),
            key=lambda x: max(e[0] for e in x[1]),
            reverse=True,
        )

        for val_id, candidates in sorted_values:
            candidates.sort(key=lambda c: c[0], reverse=True)  # по weight desc
            for weight, label_box, value_box, relation in candidates:
                if id(label_box) in used_labels:
                    continue
                used_labels.add(id(label_box))
                # Ограничиваем уверенность диапазоном [0, 1]
                pair_confidence = min(
                    weight * 100,
                    label_box.confidence * value_box.confidence,
                    1.0,  # Максимальное значение 1.0
                )
                pairs.append(
                    TextPair(
                        label=label_box,
                        value=value_box,
                        relation=relation,
                        pair_confidence=pair_confidence,
                    )
                )
                break

        return pairs

    def _detect_table_rows(self, boxes: list[TextBox]) -> list[TextPair]:
        """Детектирование табличной структуры по Y-координате.

        Строки с одинаковой Y-координатой (±tolerance):
        колонка 1 = метка, колонка 2 = значение.
        Допуск адаптируется к средней высоте текстовых блоков.
        """
        if not boxes:
            return []

        pairs: list[TextPair] = []

        # Адаптивный допуск на основе средней высоты текстовых блоков
        avg_height = sum(b.bbox.h for b in boxes) / len(boxes) if boxes else 24
        row_tolerance = max(12, int(avg_height * 0.5))

        # Группируем по Y с допуском
        sorted_boxes = sorted(boxes, key=lambda b: b.bbox.center[1])
        rows: list[list[TextBox]] = []

        current_row = [sorted_boxes[0]]
        for box in sorted_boxes[1:]:
            if abs(box.bbox.center[1] - current_row[-1].bbox.center[1]) < row_tolerance:
                current_row.append(box)
            else:
                rows.append(current_row)
                current_row = [box]
        if current_row:
            rows.append(current_row)

        for row in rows:
            if len(row) < 2:
                continue
            row_sorted = sorted(row, key=lambda b: b.bbox.center[0])
            label_box = row_sorted[0]
            value_box = row_sorted[-1]
            if label_box is value_box:
                continue
            if label_box.box_type in ("label", "unknown") and value_box.box_type in (
                "value",
                "unknown",
            ):
                pairs.append(
                    TextPair(
                        label=label_box,
                        value=value_box,
                        relation="horizontal",
                        pair_confidence=0.75,
                    )
                )

        return pairs

    def _merge_pairs(
        self,
        primary: list[TextPair],
        secondary: list[TextPair],
    ) -> list[TextPair]:
        """Объединяем две очереди пар. Приоритет — у primary.

        Args:
            primary: Основной список пар (приоритет).
            secondary: Дополнительный список пар.

        Returns:
            Объединённый список без дубликатов.
        """
        existing_labels = {id(p.label) for p in primary}
        existing_values = {id(p.value) for p in primary}
        merged = list(primary)
        for pair in secondary:
            if id(pair.label) not in existing_labels and id(pair.value) not in existing_values:
                merged.append(pair)
                existing_labels.add(id(pair.label))
                existing_values.add(id(pair.value))
        return merged

    def _merge_pairs_three(
        self,
        primary: list[TextPair],
        secondary: list[TextPair],
        tertiary: list[TextPair],
    ) -> list[TextPair]:
        """Объединяем три очереди пар. Приоритет: primary > secondary > tertiary.

        Args:
            primary: Основной список пар (column_layout, высший приоритет).
            secondary: ProximityGraph пары (средний приоритет).
            tertiary: Table row пары (низший приоритет).

        Returns:
            Объединённый список без дубликатов.
        """
        # Сначала мержим secondary + tertiary
        merged_secondary = self._merge_pairs(secondary, tertiary)
        # Затем мержим primary с результатом
        return self._merge_pairs(primary, merged_secondary)
