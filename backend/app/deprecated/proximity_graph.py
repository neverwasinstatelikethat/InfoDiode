"""Пространственный граф связей между TextBox-ами.

Использует KDTree для быстрого поиска соседей.
Строит рёбра трёх типов:
- horizontal: box_b правее box_a на той же строке
- vertical:   box_b ниже box_a в той же колонке
- inline:     уже объединены в одной строке (mixed)
"""

from __future__ import annotations

import math

import numpy as np

from app.core.ocr_models import RelationType, TextBox


class ProximityGraph:
    """Строит граф пространственных связей между TextBox-ами.

    Узлы = TextBox, рёбра = потенциальные пары (label→value).
    """

    # Максимальные расстояния для поиска соседей
    H_MAX_GAP_FACTOR = 3.0   # не более 3× высоты строки по горизонтали
    V_MAX_GAP_FACTOR = 2.0   # не более 2× высоты строки по вертикали
    ALIGN_TOLERANCE = 0.4    # допуск выравнивания (доля высоты строки)

    def build_edges(
        self, boxes: list[TextBox]
    ) -> list[tuple[TextBox, TextBox, RelationType, float]]:
        """Строит рёбра графа связей.

        Returns:
            Список (source, target, relation, weight).
            weight обратно пропорционален расстоянию (выше = ближе).
        """
        if not boxes:
            return []

        try:
            from scipy.spatial import KDTree
        except ImportError:
            # Fallback без KDTree — O(n²)
            return self._build_edges_bruteforce(boxes)

        centers = np.array([b.bbox.center for b in boxes])
        heights = np.array([b.bbox.h for b in boxes], dtype=float)

        tree = KDTree(centers)
        avg_h = float(heights.mean())
        search_radius = avg_h * max(self.H_MAX_GAP_FACTOR, self.V_MAX_GAP_FACTOR) * 2

        edges: list[tuple[TextBox, TextBox, RelationType, float]] = []
        for i, box_a in enumerate(boxes):
            cx_a, cy_a = centers[i]
            h_a = heights[i]

            idxs = tree.query_ball_point([cx_a, cy_a], r=search_radius)
            for j in idxs:
                if j == i:
                    continue
                box_b = boxes[j]
                cx_b, cy_b = centers[j]
                dx = cx_b - cx_a
                dy = cy_b - cy_a

                relation, weight = self._classify_edge(dx, dy, h_a, heights[j])
                if relation is not None:
                    edges.append((box_a, box_b, relation, weight))

        return edges

    def _build_edges_bruteforce(
        self, boxes: list[TextBox]
    ) -> list[tuple[TextBox, TextBox, RelationType, float]]:
        """Fallback: O(n²) перебор без KDTree."""
        edges: list[tuple[TextBox, TextBox, RelationType, float]] = []
        for i, box_a in enumerate(boxes):
            for j, box_b in enumerate(boxes):
                if i == j:
                    continue
                cx_a, cy_a = box_a.bbox.center
                cx_b, cy_b = box_b.bbox.center
                dx = cx_b - cx_a
                dy = cy_b - cy_a
                relation, weight = self._classify_edge(
                    dx, dy, box_a.bbox.h, box_b.bbox.h
                )
                if relation is not None:
                    edges.append((box_a, box_b, relation, weight))
        return edges

    def _classify_edge(
        self, dx: float, dy: float, h_a: float, h_b: float
    ) -> tuple[RelationType | None, float]:
        """Классифицирует ребро между двумя боксами.

        Returns:
            (relation_type, weight) или (None, 0.0) если связь недопустима.
        """
        avg_h = (h_a + h_b) / 2
        dist = math.hypot(dx, dy)

        # Горизонтальная связь: box_b правее, почти на одной высоте
        if (
            dx > 0
            and abs(dy) < avg_h * self.ALIGN_TOLERANCE
            and dx < avg_h * self.H_MAX_GAP_FACTOR
        ):
            weight = 1.0 / max(dist, 1.0)
            return "horizontal", weight

        # Вертикальная связь: box_b ниже, почти в одной колонке
        if (
            dy > 0
            and abs(dx) < avg_h * self.ALIGN_TOLERANCE
            and dy < avg_h * self.V_MAX_GAP_FACTOR
        ):
            weight = 1.0 / max(dist, 1.0)
            return "vertical", weight

        return None, 0.0
