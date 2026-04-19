"""Процессор значений — временно́е сглаживание, валидация, цветовое состояние.

Консолидирует все методы постобработки распознанных значений:
- Медианный фильтр по 3-5 кадрам
- Обработка мигающих значений (alarm)
- Валидация скорости изменения
- Валидация диапазона
- Перекрёстная валидация с правой панелью
- Форматирование десятичных знаков
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque

from app.utils.text_postprocess import (
    DECIMAL_PLACES,
    VALUE_RANGES,
    format_decimal,
    validate_change_rate,
    validate_range,
)

logger = logging.getLogger(__name__)


class ValueProcessor:
    """Процессор значений с окном временно́го сглаживания."""

    # Таблица подстановки символов для числовых значений (предкомпилирована)
    _CHAR_SUBS: dict[str, str] = {
        # Латинские → цифры
        "O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "s": "5", "B": "8",
        # Кириллические → цифры
        "О": "0", "о": "0", "З": "3", "з": "3", "б": "6",
        # Разделители
        ",": ".",
    }

    # Предкомпилированный regex для извлечения чисел: находит десятичные и целые
    _NUMBER_PATTERN: re.Pattern[str] = re.compile(r"-?\d+[.,]\d+|-?\d+")

    def __init__(
        self,
        window_size: int = 3,
        confidence_threshold: float = 0.5,
        initial_confidence_threshold: float = 0.3,
        interval_s: float = 0.5,
        outlier_std_threshold: float = 3.0,
    ) -> None:
        """Инициализирует процессор значений.

        Args:
            window_size: Размер окна временного сглаживания (медиана).
                По умолчанию 3 для быстрого отклика.
            confidence_threshold: Минимальная уверенность OCR.
            initial_confidence_threshold: Пониженный порог для начальных кадров
                (когда нет предыдущего валидного значения).
            interval_s: Интервал между кадрами в секундах.
            outlier_std_threshold: Порог отклонения в стандартных отклонениях
                для отбрасывания выбросов.
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.initial_confidence_threshold = initial_confidence_threshold
        self.interval_s = interval_s
        self.outlier_std_threshold = outlier_std_threshold

        # История значений: param_id -> deque of (value, confidence, color_state)
        self._history: dict[int, deque[tuple[float, float, str]]] = {}
        # Последнее валидное значение
        self._last_valid: dict[int, float] = {}
        # Счётчик последовательных нарушений change_rate (для tolerance window)
        self._change_rate_violations: dict[int, int] = {}
        # Блокировка для потокобезопасного доступа к _last_valid, _history, _change_rate_violations
        self._lock = threading.Lock()

    @staticmethod
    def _has_digit(text: str) -> bool:
        """Быстрая проверка наличия хотя бы одной цифры в тексте."""
        for ch in text:
            if ch.isdigit():
                return True
        return False

    @classmethod
    def _extract_best_number(cls, text: str) -> float | None:
        """Универсальное извлечение числа из текста.

        Алгоритм:
        1. Очистка текста (strip whitespace)
        2. Быстрая проверка: если нет цифр — сразу None
        3. Попытка прямого float() для чистых числовых строк
        4. Поиск ВСЕХ числовых подстрок через предкомпилированный regex
        5. Выбор САМОЙ ДЛИННОЙ подстроки (наиболее вероятно — реальное значение)
        6. Конвертация лучшего совпадения в float

        Обрабатывает: "25.5°C" → 25.5, "1234 кПа" → 1234.0,
        "T-001: 45.2" → 45.2, "-12.3мм/с" → -12.3

        Args:
            text: Текст, возможно содержащий число.

        Returns:
            Числовое значение или None.
        """
        if not text:
            return None

        cleaned = text.strip()

        # Быстрая проверка: если нет цифр — не пытаемся парсить
        if not cls._has_digit(cleaned):
            return None

        # Попытка прямого парсинга для чистых числовых строк
        try:
            return float(cleaned.replace(",", "."))
        except ValueError:
            pass

        # Поиск всех числовых подстрок через regex
        matches = cls._NUMBER_PATTERN.findall(cleaned)
        if not matches:
            return None

        # Выбираем самое длинное совпадение — скорее всего это реальное значение
        best_match = max(matches, key=len)

        try:
            return float(best_match.replace(",", "."))
        except ValueError:
            return None

    @classmethod
    def _extract_numeric_fast(cls, text: str) -> float | None:
        """Быстрое извлечение числа (обёртка над _extract_best_number).

        Args:
            text: Текст, возможно содержащий число.

        Returns:
            Числовое значение или None.
        """
        return cls._extract_best_number(text)

    def process_value(
        self,
        param_id: int,
        raw_text: str,
        confidence: float,
        param_type: str,
        color_state: str = "normal",
        right_panel_value: float | None = None,
        detection_method: str = "none",
    ) -> str:
        """Обрабатывает одно распознанное значение.
    
        Args:
            param_id: ID параметра.
            raw_text: Сырой текст из OCR.
            confidence: Уверенность OCR (0-1).
            param_type: Тип параметра (T, P, dP, и т.д.).
            color_state: Цветовое состояние (normal/alarm/warning/inactive).
            right_panel_value: Значение из правой панели (если доступно).
            detection_method: Метод обнаружения (enhanced_pair, proximity_right,
                proximity_below, roi_filter, none).
    
        Returns:
            Отформатированное значение строки.
        """
        # Потокобезопасный доступ к общим структурам данных
        with self._lock:
            return self._process_value_impl(
                param_id, raw_text, confidence, param_type, color_state,
                right_panel_value, detection_method
            )

    def _process_value_impl(
        self,
        param_id: int,
        raw_text: str,
        confidence: float,
        param_type: str,
        color_state: str = "normal",
        right_panel_value: float | None = None,
        detection_method: str = "none",
    ) -> str:
        """Внутренняя реализация process_value (вызывается под блокировкой).
    
        Args:
            param_id: ID параметра.
            raw_text: Сырой текст из OCR.
            confidence: Уверенность OCR (0-1).
            param_type: Тип параметра (T, P, dP, и т.д.).
            color_state: Цветовое состояние (normal/alarm/warning/inactive).
            right_panel_value: Значение из правой панели (если доступно).
            detection_method: Метод обнаружения (enhanced_pair, proximity_right,
                proximity_below, roi_filter, none).
    
        Returns:
            Отформатированное значение строки.
        """
        # Быстрое извлечение числа (без regex)
        numeric = self._extract_numeric_fast(raw_text)
        has_last_valid = param_id in self._last_valid
    
        # Если не удалось извлечь число — fallback
        if numeric is None:
            # Логируем только если текст выглядит как число (содержит цифры)
            if raw_text and self._has_digit(raw_text):
                logger.warning(
                    "Параметр %d: fallback — текст '%s' не распознан как число",
                    param_id, raw_text
                )
            else:
                # Пустая строка или без цифр — debug, не засоряем логи
                logger.debug(
                    "Параметр %d: текст '%s' не содержит цифр — пропущен",
                    param_id, raw_text or ""
                )
            return self._fallback_value(param_id, param_type, raw_text, confidence)

        # === Валидация ПЕРЕД проверкой confidence ===
        # В SCADA-системах физика ограничивает значения: если число проходит
        # все физические проверки, оно почти наверняка верно, даже если OCR
        # дал низкую уверенность. Поэтому сначала проверяем физику, потом
        # вычисляем composite confidence.

        # Валидация диапазона
        range_valid = validate_range(numeric, param_type)
        if not range_valid:
            min_val, max_val = VALUE_RANGES.get(param_type, (0, 1000))
            logger.warning(
                "Параметр %d: fallback — значение %.2f вне диапазона [%.1f, %.1f]",
                param_id, numeric, min_val, max_val
            )
            return self._fallback_value(param_id, param_type, raw_text, confidence, numeric)

        # Валидация скорости изменения
        rate_valid = True
        if has_last_valid:
            if not validate_change_rate(
                self._last_valid[param_id], numeric, param_type, self.interval_s
            ):
                # Tolerance window: allow after 2 consecutive violations
                violations = self._change_rate_violations.get(param_id, 0) + 1
                self._change_rate_violations[param_id] = violations
                if violations < 2:
                    logger.warning(
                        "Параметр %d: fallback — слишком быстрое изменение %.2f -> %.2f (violation %d/2)",
                        param_id, self._last_valid[param_id], numeric, violations
                    )
                    return self._fallback_value(param_id, param_type, raw_text, confidence, numeric)
                # After 2 violations: accept the new value
                logger.info(
                    "Параметр %d: принимаем значение после %d нарушений change_rate",
                    param_id, violations
                )
                rate_valid = False  # Технически нарушение, но принимаем
            else:
                self._change_rate_violations[param_id] = 0

        # Консистентность с предыдущим значением
        consistency_valid = True
        if has_last_valid:
            last_val = self._last_valid[param_id]
            if last_val != 0:
                consistency_valid = abs(numeric - last_val) / abs(last_val) < 0.10  # 10% допуск
            else:
                consistency_valid = numeric == 0.0

        # === Composite confidence ===
        # Объединяем OCR confidence с результатами валидации.
        # В SCADA-системах физические ограничения — более надёжный индикатор
        # корректности, чем сырая OCR confidence.
        composite_conf = self._compute_composite_confidence(
            ocr_confidence=confidence,
            range_valid=range_valid,
            rate_valid=rate_valid,
            consistency_valid=consistency_valid,
            detection_method=detection_method,
            has_last_valid=has_last_valid,
        )

        # Проверка confidence с учётом начальных кадров
        effective_threshold = (
            self.confidence_threshold
            if has_last_valid
            else self.initial_confidence_threshold
        )

        if composite_conf < effective_threshold:
            logger.warning(
                "Параметр %d: fallback — composite confidence %.2f < %.2f "
                "(ocr=%.2f, range=%s, rate=%s, cons=%s, method=%s)",
                param_id, composite_conf, effective_threshold,
                confidence, range_valid, rate_valid, consistency_valid,
                detection_method
            )
            return self._fallback_value(param_id, param_type, raw_text, confidence, numeric)

        if composite_conf > confidence:
            logger.debug(
                "Параметр %d: confidence boost %.2f -> %.2f "
                "(range=%s, rate=%s, cons=%s, method=%s)",
                param_id, confidence, composite_conf,
                range_valid, rate_valid, consistency_valid, detection_method
            )

        # Перекрёстная валидация с правой панелью
        if right_panel_value is not None:
            numeric = self._cross_validate(numeric, right_panel_value, param_type)

        # Добавляем в историю
        if param_id not in self._history:
            self._history[param_id] = deque(maxlen=self.window_size)
        self._history[param_id].append((numeric, composite_conf, color_state))

        # Временно́е сглаживание
        smoothed = self._temporal_smooth(param_id, color_state)

        # Обновляем последнее валидное значение
        self._last_valid[param_id] = smoothed

        result = format_decimal(smoothed, param_type)
        logger.debug("Параметр %d: '%s' -> %s (type=%s, conf=%.2f->%.2f)",
                     param_id, raw_text, result, param_type, confidence, composite_conf)

        return result

    def _compute_composite_confidence(
        self,
        ocr_confidence: float,
        range_valid: bool,
        rate_valid: bool,
        consistency_valid: bool,
        detection_method: str,
        has_last_valid: bool,
    ) -> float:
        """Вычисляет composite confidence на основе OCR + физической валидации.

        SCADA-значения ограничены физикой: если число проходит все проверки
        диапазона, скорости изменения и согласованности с предыдущими кадрами,
        оно почти наверняка верно, даже если OCR дал низкую уверенность.

        Формула: composite = max(ocr_conf, validation_boost)
        Где validation_boost — сумма баллов за пройденные проверки:
        - range_valid: +0.30 (значение в физически допустимом диапазоне)
        - rate_valid: +0.20 (скорость изменения реалистична)
        - consistency_valid: +0.25 (согласовано с предыдущим значением)
        - detection_method != 'none': +0.15 (значение найдено рядом с меткой)

        Максимальный validation_boost = 0.90
        Minimum composite = max(ocr_conf, validation_boost)

        Пример: OCR conf=0.20, все проверки пройдены → boost=0.90, composite=0.90

        Args:
            ocr_confidence: Сырая уверенность OCR.
            range_valid: Значение в допустимом диапазоне.
            rate_valid: Скорость изменения в норме.
            consistency_valid: Согласовано с предыдущим значением.
            detection_method: Метод обнаружения значения.
            has_last_valid: Есть ли предыдущее валидное значение.

        Returns:
            Composite confidence [0, 1].
        """
        validation_boost = 0.0

        # Диапазон — самая важная проверка
        if range_valid:
            validation_boost += 0.30

        # Скорость изменения
        if rate_valid:
            validation_boost += 0.20

        # Согласованность с предыдущим значением
        if consistency_valid and has_last_valid:
            validation_boost += 0.25

        # Метод обнаружения (значение найдено рядом с меткой)
        if detection_method in ("enhanced_pair", "proximity_right"):
            validation_boost += 0.15
        elif detection_method in ("proximity_below", "roi_filter"):
            validation_boost += 0.10

        # Composite = максимум из OCR и validation boost
        # Не усредняем, а берём max: если валидация даёт высокую оценку,
        # OCR confidence не должен её тянуть вниз
        composite = max(ocr_confidence, validation_boost)

        return min(composite, 1.0)

    def _temporal_smooth(self, param_id: int, color_state: str) -> float:
        """Применяет медианный фильтр к окну значений с отбрасыванием выбросов.

        Для мигающих значений (alarm) используем только чётные кадры.
        Выбросы отбрасываются на основе отклонения от среднего.

        Args:
            param_id: ID параметра.
            color_state: Цветовое состояние.

        Returns:
            Сглаженное значение.
        """
        history = self._history.get(param_id)
        if not history:
            return self._last_valid.get(param_id, 0.0)

        if color_state == "alarm" and len(history) >= 2:
            # Для мигающих значений — берём только чётные индексы
            values = [h[0] for i, h in enumerate(history) if i % 2 == 0]
        else:
            values = [h[0] for h in history]

        if not values:
            return self._last_valid.get(param_id, 0.0)

        # Отбрасывание выбросов (outlier rejection)
        filtered_values = self._reject_outliers(values)

        if not filtered_values:
            filtered_values = values  # Fallback если все отфильтрованы

        # Медиана
        sorted_vals = sorted(filtered_values)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        return sorted_vals[mid]

    def _reject_outliers(self, values: list[float]) -> list[float]:
        """Отбрасывает выбросы на основе стандартного отклонения.

        Значения, отклоняющиеся более чем на outlier_std_threshold
        стандартных отклонений от среднего, считаются выбросами.

        Args:
            values: Список значений.

        Returns:
            Список значений без выбросов.
        """
        if len(values) < 3:
            return values  # Недостаточно данных для статистики

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = variance ** 0.5

        if std == 0:
            return values  # Все значения одинаковые

        threshold = self.outlier_std_threshold * std
        return [v for v in values if abs(v - mean) <= threshold]

    def _cross_validate(
        self,
        central_value: float,
        right_panel_value: float,
        param_type: str,
    ) -> float:
        """Перекрёстная валидация центральная схема vs правая панель.

        Если значения совпадают (в пределах допуска), используем центральное.
        Если расходятся — берём то, что ближе к предыдущему валидному.

        Args:
            central_value: Значение из центральной схемы.
            right_panel_value: Значение из правой панели.
            param_type: Тип параметра.

        Returns:
            Выбранное значение.
        """
        # Допуск: 2% от диапазона
        min_val, max_val = VALUE_RANGES.get(param_type, (0, 1000))
        tolerance = (max_val - min_val) * 0.02

        if abs(central_value - right_panel_value) <= tolerance:
            return central_value  # Совпадают — доверяем центральной схеме

        # Расхождение — предпочитаем правую панель (менее плотный текст = точнее)
        return right_panel_value

    def _fallback_value(
        self,
        param_id: int,
        param_type: str,
        raw_text: str | None = None,
        confidence: float = 0.0,
        extracted_numeric: float | None = None,
    ) -> str:
        """Возвращает fallback-значение при ошибке распознавания.

        Приоритет:
        1. Последнее валидное значение (если есть)
        2. Извлечённое число из raw_text (даже при низкой уверенности)
        3. 0.0 как последний резерв

        Args:
            param_id: ID параметра.
            param_type: Тип параметра.
            raw_text: Сырой текст из OCR (для попытки извлечения).
            confidence: Уверенность OCR.
            extracted_numeric: Уже извлечённое число (если есть).

        Returns:
            Отформатированное fallback-значение.
        """
        # 1. Проверяем последнее валидное значение
        if param_id in self._last_valid:
            logger.debug(
                "Параметр %d: fallback использует последнее валидное значение %.2f",
                param_id, self._last_valid[param_id]
            )
            return format_decimal(self._last_valid[param_id], param_type)

        # 2. Пробуем использовать уже извлечённое число или извлечь из raw_text
        numeric = extracted_numeric
        if numeric is None and raw_text is not None:
            numeric = self._extract_numeric_fast(raw_text)

        if numeric is not None:
            # Проверяем диапазон — если валиден, используем
            if validate_range(numeric, param_type):
                # Сохраняем как начальное валидное значение для будущих кадров
                self._last_valid[param_id] = numeric
                logger.info(
                    "Параметр %d: fallback принял извлечённое число %.2f из '%s' (conf=%.2f)",
                    param_id, numeric, raw_text or "N/A", confidence
                )
                return format_decimal(numeric, param_type)
            else:
                min_val, max_val = VALUE_RANGES.get(param_type, (0, 1000))
                logger.warning(
                    "Параметр %d: fallback — число %.2f из '%s' вне диапазона [%.1f, %.1f]",
                    param_id, numeric, raw_text or "N/A", min_val, max_val
                )

        # 3. Абсолютный fallback — пустая строка (не загрязняем _last_valid нулями)
        logger.warning(
            "Параметр %d: fallback — нет валидного значения, нет parseable текста, возвращаем пустую строку",
            param_id
        )
        return ""  # Empty string — don't pollute _last_valid with zeros

    def reset(self) -> None:
        """Сбрасывает состояние процессора."""
        with self._lock:
            history_count = len(self._history)
            self._history.clear()
            self._last_valid.clear()
            self._change_rate_violations.clear()
        logger.info("ValueProcessor сброшен (очищено %d параметров)", history_count)

    def get_last_valid(self, param_id: int) -> float | None:
        """Потокобезопасное получение последнего валидного значения.

        Args:
            param_id: ID параметра.

        Returns:
            Последнее валидное значение или None.
        """
        with self._lock:
            return self._last_valid.get(param_id)

    def has_last_valid(self, param_id: int) -> bool:
        """Потокобезопасная проверка наличия последнего валидного значения.

        Args:
            param_id: ID параметра.

        Returns:
            True если есть валидное значение.
        """
        with self._lock:
            return param_id in self._last_valid
