"""Скоринг уверенности для OCR-результатов с мультимодальной фьюзией.

Учитывает:
- Базовый confidence распознавания текста
- Цветовые признаки (положение внутри индикатора)
- Физический диапазон параметра
- Временную стабильность значений
- Временное сглаживание для фильтрации выбросов
"""

from __future__ import annotations

import logging
import re

from app.core.ocr_models import TextBox, TextPair

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Вычисляет финальный confidence для TextBox с учётом контекста.

    Использует мультимодальную фьюзию: OCR confidence, цветовые признаки,
    физическая валидность и временная стабильность.
    Также реализует временное сглаживание для фильтрации выбросов.
    """

    # Веса для компонентов скоринга
    BASE_WEIGHT: float = 0.7  # Базовый вес OCR confidence
    COLOR_BONUS_INSIDE: float = 0.15  # Бонус для значений внутри индикаторов
    COLOR_PENALTY_UNKNOWN: float = -0.3  # Штраф для чисел без цветового контекста
    RANGE_BONUS: float = 0.1  # Бонус за попадание в диапазон параметра
    TEMPORAL_BONUS: float = 0.1  # Бонус за временную стабильность

    # Порог для временной стабильности (10% отклонение от медианы)
    TEMPORAL_THRESHOLD: float = 0.1

    # Порог для временного сглаживания (50% отклонение от среднего)
    SMOOTHING_THRESHOLD: float = 0.5
    # Максимальный размер истории значений
    MAX_HISTORY_SIZE: int = 5

    def __init__(self) -> None:
        """Инициализирует скорер с пустой историей значений."""
        # История значений: param_id -> list[(value, confidence)]
        self._value_history: dict[int, list[tuple[str, float]]] = {}

    def smooth_value(
        self,
        param_id: int,
        current_value: str,
        current_confidence: float,
    ) -> tuple[str, float]:
        """Применяет временное сглаживание для фильтрации выбросов.

        Сохраняет историю значений (макс. 5 записей) и отклоняет
        значения, сильно отличающиеся от скользящего среднего.

        Args:
            param_id: Идентификатор параметра.
            current_value: Текущее распознанное значение.
            current_confidence: Уверенность текущего значения.

        Returns:
            Кортеж (value, confidence) — отфильтрованное значение.
        """
        # Инициализируем историю для параметра
        if param_id not in self._value_history:
            self._value_history[param_id] = []

        history = self._value_history[param_id]

        # Добавляем текущее значение в историю (FIFO, макс. 5 записей)
        history.append((current_value, current_confidence))
        if len(history) > self.MAX_HISTORY_SIZE:
            history.pop(0)

        # Если история слишком короткая — возвращаем как есть
        if len(history) < 2:
            return (current_value, current_confidence)

        # Пытаемся распарсить текущее значение как число
        current_numeric = self._parse_numeric_value(current_value)
        if current_numeric is None:
            # Не числовое значение — возвращаем как есть
            return (current_value, current_confidence)

        # Вычисляем скользящее среднее по числовым значениям истории
        numeric_values: list[float] = []
        for val, _ in history:
            parsed = self._parse_numeric_value(val)
            if parsed is not None:
                numeric_values.append(parsed)

        if len(numeric_values) < 2:
            return (current_value, current_confidence)

        rolling_avg = sum(numeric_values) / len(numeric_values)

        # Проверяем отклонение от среднего
        if rolling_avg == 0:
            deviation = abs(current_numeric)
        else:
            deviation = abs(current_numeric - rolling_avg) / abs(rolling_avg)

        # Если отклонение > 50% и уверенность < 0.85 — отклоняем как выброс
        if deviation > self.SMOOTHING_THRESHOLD and current_confidence < 0.85:
            # Возвращаем предыдущее значение
            prev_value, prev_confidence = history[-2]
            logger.debug(
                "Temporal smoothing: rejected outlier %s for param %d (avg=%.1f, conf=%.2f)",
                current_value, param_id, rolling_avg, current_confidence
            )
            return (prev_value, prev_confidence)

        return (current_value, current_confidence)

    def reset_history(self, param_id: int | None = None) -> None:
        """Сбрасывает историю значений.

        Args:
            param_id: ID параметра для сброса. Если None — сбрасывает всю историю.
        """
        if param_id is None:
            self._value_history.clear()
        else:
            self._value_history.pop(param_id, None)

    def score(
        self,
        text_box: TextBox,
        pair: TextPair | None = None,
        param_range: tuple[float, float] | None = None,
        recent_values: list[float] | None = None,
    ) -> float:
        """Вычисляет итоговый confidence score для TextBox.

        Применяет мультимодальную фьюзию: базовый OCR confidence,
        цветовые бонусы/штрафы, проверку диапазона и временную стабильность.

        Args:
            text_box: Блок текста с OCR confidence и метаданными.
            pair: Опциональная пара label:value для контекста.
            param_range: Опциональный кортеж (min, max) допустимых значений.
            recent_values: Опциональный список недавних значений для проверки стабильности.

        Returns:
            Итоговый confidence в диапазоне [0.0, 1.0].
        """
        # Базовый score от OCR confidence
        final_score = text_box.confidence * self.BASE_WEIGHT

        # Цветовой бонус/штраф на основе color_tag
        color_adjustment = self._compute_color_adjustment(text_box)
        final_score += color_adjustment

        # Бонус за попадание в физический диапазон параметра
        if param_range is not None:
            range_bonus = self._compute_range_bonus(text_box, param_range)
            final_score += range_bonus

        # Бонус за временную стабильность
        if recent_values is not None and len(recent_values) > 0:
            temporal_bonus = self._compute_temporal_bonus(text_box, recent_values)
            final_score += temporal_bonus

        # Клэмпинг в диапазон [0.0, 1.0]
        return max(0.0, min(1.0, final_score))

    def _compute_color_adjustment(self, text_box: TextBox) -> float:
        """Вычисляет цветовую корректировку на основе color_tag.

        Args:
            text_box: Блок текста с полем box_type или source.

        Returns:
            Корректировка score (положительная или отрицательная).
        """
        # Определяем эффективный color_tag: используем color_tag если он задан,
        # иначе используем box_type как fallback
        color_tag = getattr(text_box, 'color_tag', 'unknown')
        if color_tag == 'unknown':
            # Используем box_type как fallback если color_tag не задан
            color_tag = text_box.box_type

        # Бонус для значений внутри цветных индикаторов
        if color_tag and color_tag.startswith("value_"):
            return self.COLOR_BONUS_INSIDE

        # Проверяем, является ли текст числовым значением
        is_numeric = self._is_numeric_value(text_box.text)

        # Штраф для числовых значений без цветового контекста
        if is_numeric and (not color_tag or color_tag in ("unknown", "label")):
            return self.COLOR_PENALTY_UNKNOWN

        return 0.0

    def _compute_range_bonus(
        self,
        text_box: TextBox,
        param_range: tuple[float, float],
    ) -> float:
        """Вычисляет бонус за попадание значения в допустимый диапазон.

        Args:
            text_box: Блок текста с распознанным значением.
            param_range: Кортеж (min, max) допустимых значений.

        Returns:
            Бонус score если значение в диапазоне, иначе 0.
        """
        parsed_value = self._parse_numeric_value(text_box.text)

        if parsed_value is None:
            return 0.0

        min_val, max_val = param_range

        if min_val <= parsed_value <= max_val:
            return self.RANGE_BONUS

        return 0.0

    def _compute_temporal_bonus(
        self,
        text_box: TextBox,
        recent_values: list[float],
    ) -> float:
        """Вычисляет бонус за временную стабильность значения.

        Значение считается стабильным, если отклонение от медианы
        не превышает 10%.

        Args:
            text_box: Блок текста с текущим значением.
            recent_values: Список недавних значений параметра.

        Returns:
            Бонус score если значение стабильно, иначе 0.
        """
        if not recent_values:
            return 0.0

        current_value = self._parse_numeric_value(text_box.text)

        if current_value is None:
            return 0.0

        # Вычисляем медиану недавних значений
        sorted_values = sorted(recent_values)
        n = len(sorted_values)

        if n == 0:
            return 0.0

        if n % 2 == 1:
            median = sorted_values[n // 2]
        else:
            median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2

        # Проверяем отклонение от медианы (10% порог)
        if median == 0:
            deviation = abs(current_value)
        else:
            deviation = abs(current_value - median) / abs(median)

        if deviation <= self.TEMPORAL_THRESHOLD:
            return self.TEMPORAL_BONUS

        return 0.0

    def _is_numeric_value(self, text: str) -> bool:
        """Проверяет, является ли текст числовым значением.

        Args:
            text: Распознанный текст.

        Returns:
            True если текст содержит число, иначе False.
        """
        if not text:
            return False

        # Убираем пробелы и типичные единицы измерения
        cleaned = text.strip()
        cleaned = re.sub(r'[\s°%МПаkPaмм/сVHz]', '', cleaned)
        cleaned = re.sub(r'[,]', '.', cleaned)  # Заменяем запятую на точку

        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    def _parse_numeric_value(self, text: str) -> float | None:
        """Извлекает числовое значение из текста.

        Args:
            text: Распознанный текст с возможными единицами измерения.

        Returns:
            Числовое значение как float, или None если не удалось распарсить.
        """
        if not text:
            return None

        # Убираем пробелы и единицы измерения
        cleaned = text.strip()

        # Убираем типичные единицы измерения SCADA
        units = [
            '°C', '°', 'deg C', 'degC',
            'МПа', 'MPa', 'kPa', 'Pa',
            'мм/с', 'mm/s',
            'мм', 'mm', '%',
            'Hz', 'В', 'V',
            'units', 'ед',
        ]

        for unit in units:
            cleaned = cleaned.replace(unit, '')
            cleaned = cleaned.replace(unit.lower(), '')

        # Заменяем запятую на точку для десятичных чисел
        cleaned = cleaned.replace(',', '.')

        # Убираем оставшиеся пробелы
        cleaned = cleaned.strip()

        try:
            return float(cleaned)
        except ValueError:
            return None

    def score_batch(
        self,
        text_boxes: list[TextBox],
        pairs: list[TextPair] | None = None,
        param_ranges: dict[str, tuple[float, float]] | None = None,
        recent_values_map: dict[str, list[float]] | None = None,
    ) -> list[float]:
        """Вычисляет confidence scores для списка TextBox.

        Args:
            text_boxes: Список блоков текста.
            pairs: Опциональный список пар для контекста.
            param_ranges: Опциональный словарь диапазонов по ID параметров.
            recent_values_map: Опциональный словарь недавних значений.

        Returns:
            Список итоговых confidence scores.
        """
        scores: list[float] = []

        for text_box in text_boxes:
            # Ищем соответствующую пару
            pair = None
            if pairs:
                for p in pairs:
                    if p.label == text_box or p.value == text_box:
                        pair = p
                        break

            # Получаем диапазон для параметра
            param_range = None
            if param_ranges and text_box.text in param_ranges:
                param_range = param_ranges[text_box.text]

            # Получаем недавние значения
            recent_values = None
            if recent_values_map and text_box.text in recent_values_map:
                recent_values = recent_values_map[text_box.text]

            score = self.score(text_box, pair, param_range, recent_values)
            scores.append(score)

        return scores
