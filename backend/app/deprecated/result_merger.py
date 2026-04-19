"""Слияние результатов Paddle + Florence.

Стратегия:
- Paddle — основа (всегда актуален, быстрый)
- Florence — корректор: исправляет label-ы где Paddle ошибся
- Если Florence нашёл параметр, которого нет у Paddle — добавляем

Fuzzy-матчинг через Levenshtein для alignment label-ов.
Добавлен bbox IoU как вторичный критерий matching.
"""

from __future__ import annotations

import logging

from app.core.ocr_models import BBox, SourceType, TextBox, TextPair

logger = logging.getLogger(__name__)

# Пороги для matching
TEXT_MATCH_THRESHOLD = 70  # Fuzzy text match
TEXT_MATCH_STRONG = 90  # Сильный text match (IoU не требуется)
BBOX_IOU_THRESHOLD = 0.3  # Минимальный IoU для spatial matching


class ResultMerger:
    """Объединяет результаты Paddle (актуальные, быстрые) и Florence (семантические)."""

    @staticmethod
    def _compute_iou(box1: BBox, box2: BBox) -> float:
        """Вычисляет Intersection over Union (IoU) двух bounding box.

        Args:
            box1: Первый bounding box.
            box2: Второй bounding box.

        Returns:
            IoU в диапазоне [0.0, 1.0].
        """
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.w, box2.x + box2.w)
        y2 = min(box1.y + box1.h, box2.y + box2.h)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.w * box1.h
        area2 = box2.w * box2.h
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def merge(
        self,
        paddle_pairs: list[TextPair],
        florence_pairs: list[TextPair] | None,
    ) -> tuple[dict[str, str], list[TextPair]]:
        """Сливает два набора пар.

        Args:
            paddle_pairs: Пары от Paddle (основа).
            florence_pairs: Пары от Florence (корректор) или None.

        Returns:
            (raw_fields, merged_pairs).
        """
        if not florence_pairs:
            return self._pairs_to_dict(paddle_pairs), paddle_pairs

        merged_pairs = list(paddle_pairs)
        paddle_labels = {p.label.text: p for p in paddle_pairs}

        for fl_pair in florence_pairs:
            fl_label = fl_pair.label.text

            # Ищем соответствие в paddle по fuzzy matching + bbox IoU
            best_match, best_score, iou = self._find_best_match_with_iou(
                fl_pair, paddle_labels
            )

            # Strong match: text > 90 (текст достаточно, IoU не важен)
            # Normal match: text > 70 AND IoU > 0.3
            is_strong_match = best_score >= TEXT_MATCH_STRONG
            is_normal_match = (
                best_score >= TEXT_MATCH_THRESHOLD and iou >= BBOX_IOU_THRESHOLD
            )

            if is_strong_match or is_normal_match:
                # Florence подтвердил параметр — повышаем уверенность
                pd_pair = paddle_labels[best_match]
                pd_pair.pair_confidence = min(1.0, pd_pair.pair_confidence * 1.1)
                # Если Florence дала лучший label — заменяем
                if len(fl_label) > len(pd_pair.label.text):
                    pd_pair.label.text = fl_label
            else:
                # Florence нашёл новый параметр — добавляем
                merged_pairs.append(fl_pair)
                logger.debug(
                    "Florence добавила новый параметр: %s = %s",
                    fl_label,
                    fl_pair.value.text,
                )

        return self._pairs_to_dict(merged_pairs), merged_pairs

    def _find_best_match_with_iou(
        self,
        fl_pair: TextPair,
        candidates: dict[str, TextPair],
    ) -> tuple[str, float, float]:
        """Ищет лучший fuzzy-матч для Florence пары среди candidates с учётом IoU.

        Args:
            fl_pair: Пара от Florence (содержит label и value боксы).
            candidates: Словарь label -> TextPair от Paddle.

        Returns:
            Кортеж (best_candidate_key, best_text_score, iou_with_best).
        """
        if not candidates:
            return "", 0.0, 0.0

        query = fl_pair.label.text
        fl_bbox = fl_pair.label.bbox

        try:
            from rapidfuzz import fuzz

            best_match = ""
            best_score = 0.0
            best_iou = 0.0

            for candidate_label, candidate_pair in candidates.items():
                # Text similarity
                text_score = fuzz.token_sort_ratio(query, candidate_label)

                # Bbox IoU (если доступны координаты)
                iou = 0.0
                if candidate_pair.label.bbox and fl_bbox:
                    iou = self._compute_iou(fl_bbox, candidate_pair.label.bbox)

                # Обновляем лучший match
                if text_score > best_score:
                    best_score = text_score
                    best_match = candidate_label
                    best_iou = iou

            return best_match, best_score, best_iou

        except ImportError:
            # Fallback: Levenshtein distance
            from Levenshtein import distance as levenshtein_distance

            best_match = ""
            best_score = 0.0
            best_iou = 0.0

            for candidate_label, candidate_pair in candidates.items():
                dist = levenshtein_distance(query.lower(), candidate_label.lower())
                max_len = max(len(query), len(candidate_label), 1)
                text_score = (1 - dist / max_len) * 100

                # Bbox IoU
                iou = 0.0
                if candidate_pair.label.bbox and fl_bbox:
                    iou = self._compute_iou(fl_bbox, candidate_pair.label.bbox)

                if text_score > best_score:
                    best_score = text_score
                    best_match = candidate_label
                    best_iou = iou

            return best_match, best_score, best_iou

    def _find_best_match(
        self,
        query: str,
        candidates: dict[str, TextPair],
    ) -> tuple[str, float]:
        """Ищет лучший fuzzy-матч для query среди candidates (legacy).

        Returns:
            (best_candidate_key, best_score).
        """
        best_match, best_score, _ = self._find_best_match_with_iou(
            TextPair(label=TextBox(bbox=BBox(x=0, y=0, w=0, h=0), text=query, confidence=0.0, source="florence"),
                     value=TextBox(bbox=BBox(x=0, y=0, w=0, h=0), text="", confidence=0.0, source="florence"),
                     pair_confidence=0.0),
            candidates
        )
        return best_match, best_score

    @staticmethod
    def _pairs_to_dict(pairs: list[TextPair]) -> dict[str, str]:
        """Конвертирует список пар в словарь {label: value}."""
        result: dict[str, str] = {}
        for pair in pairs:
            label = pair.label.text.strip()
            value = pair.value.text.strip()
            if label and value:
                result[label] = value
        return result
