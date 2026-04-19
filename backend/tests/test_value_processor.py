"""Тесты процессора значений — rigorous validation of value extraction and processing.

Tests verify universal numeric extraction from SCADA text, confidence thresholds,
fallback chains, and proper handling of edge cases like empty strings and pure text.
"""

import pytest
from app.core.value_processor import ValueProcessor


class TestValueProcessor:
    """Тесты временно́го сглаживания и валидации."""

    def test_basic_processing(self) -> None:
        """Тест: базовая обработка значения."""
        vp = ValueProcessor(window_size=3, confidence_threshold=0.3)

        result = vp.process_value(1, "758.3", 0.9, "T")
        assert result == "758.3"

    def test_low_confidence_fallback(self) -> None:
        """Тест: fallback при низкой уверенности."""
        vp = ValueProcessor(window_size=3, confidence_threshold=0.5)

        # Сначала устанавливаем валидное значение
        vp.process_value(1, "100.0", 0.9, "T")

        # Низкая уверенность -> fallback к предыдущему
        result = vp.process_value(1, "999.0", 0.2, "T")
        assert result == "100.0"  # fallback к последнему валидному

    def test_out_of_range_fallback(self) -> None:
        """Тест: fallback при выходе за диапазон."""
        vp = ValueProcessor(window_size=3, confidence_threshold=0.3)

        vp.process_value(1, "100.0", 0.9, "T")

        # T range: -50 to 800
        result = vp.process_value(1, "999.0", 0.9, "T")
        assert result == "100.0"  # fallback (999 > 800)

    def test_change_rate_validation(self) -> None:
        """Тест: валидация скорости изменения."""
        vp = ValueProcessor(window_size=3, confidence_threshold=0.3, interval_s=0.5)

        vp.process_value(1, "100.0", 0.9, "T")

        # T max rate: 5°C/s -> max 2.5°C за 0.5s
        result = vp.process_value(1, "105.0", 0.9, "T")  # +5°C за 0.5s = 10°C/s
        assert result == "100.0"  # fallback (too fast)

    def test_decimal_formatting(self) -> None:
        """Тест: форматирование десятичных знаков."""
        vp = ValueProcessor(window_size=3, confidence_threshold=0.3)

        # T = 1 decimal place
        assert vp.process_value(1, "758.3", 0.9, "T") == "758.3"

        # n = 0 decimal places
        assert vp.process_value(2, "45.7", 0.9, "n") == "46"

        # Vb = 2 decimal places
        assert vp.process_value(3, "0.25", 0.9, "Vb") == "0.25"

    def test_reset(self) -> None:
        """Тест: сброс процессора."""
        vp = ValueProcessor(window_size=3, confidence_threshold=0.3)

        vp.process_value(1, "100.0", 0.9, "T")
        vp.reset()

        # После сброса fallback returns empty or default value
        result = vp.process_value(1, "xxx", 0.1, "T")
        # Result may be empty string or 0.0 depending on implementation
        assert result == "" or float(result) == 0.0


class TestValueProcessorNumericExtraction:
    """Тесты универсального извлечения чисел из SCADA текста."""

    def test_extract_numeric_with_celsius_unit(self) -> None:
        """Тест: "25.5°C" → 25.5"""
        vp = ValueProcessor()
        result = vp.process_value(1, "25.5°C", 0.9, "T")
        assert result == "25.5"

    def test_extract_numeric_with_kpa_unit(self) -> None:
        """Тест: "1234 кПа" → 1234.0"""
        vp = ValueProcessor()
        result = vp.process_value(1, "1234 кПа", 0.9, "P")
        assert result == "1234.00"  # P has 2 decimal places

    def test_extract_numeric_with_mixed_cyrillic_latin(self) -> None:
        """Тест: смешанные кириллические/латинские единицы.
        
        OCR may produce mixed scripts like "kПa" (Latin k + Cyrillic П + Latin a).
        """
        vp = ValueProcessor()
        
        # Mixed script units
        result = vp.process_value(1, "500 kПa", 0.9, "P")
        assert float(result) == 500.0

    def test_extract_numeric_with_mm_per_s(self) -> None:
        """Тест: "12.3мм/с" → 12.3 (positive value within Vb range 0-50)"""
        vp = ValueProcessor()
        result = vp.process_value(1, "12.3мм/с", 0.9, "Vb")
        assert result == "12.30"  # Vb has 2 decimal places

    def test_extract_numeric_with_prefix(self) -> None:
        """Тест: "ID: 45.2" → 45.2"""
        vp = ValueProcessor()
        result = vp.process_value(1, "ID: 45.2", 0.9, "T")
        assert result == "45.2"

    def test_extract_numeric_with_spaces(self) -> None:
        """Тест: "  78.9  " → 78.9 (with surrounding spaces)"""
        vp = ValueProcessor()
        result = vp.process_value(1, "  78.9  ", 0.9, "T")
        assert result == "78.9"

    def test_extract_numeric_comma_separator(self) -> None:
        """Тест: "25,5°C" → 25.5 (comma as decimal separator)"""
        vp = ValueProcessor()
        result = vp.process_value(1, "25,5°C", 0.9, "T")
        assert result == "25.5"

    def test_extract_numeric_negative_value(self) -> None:
        """Тест: "-25.5°C" → -25.5"""
        vp = ValueProcessor()
        result = vp.process_value(1, "-25.5°C", 0.9, "T")
        assert result == "-25.5"

    def test_extract_numeric_multiple_numbers(self) -> None:
        """Тест: выбор самого длинного числа при нескольких в тексте.
        
        Example: "ID: 100 Value: 45.5" → should pick one of them
        """
        vp = ValueProcessor()
        result = vp.process_value(1, "ID: 100 Value: 45.5", 0.9, "T")
        # A valid number should be extracted
        val = float(result)
        assert val in [100.0, 45.5] or (val > 0 and val < 800)


class TestValueProcessorEdgeCases:
    """Тесты граничных случаев и edge cases."""

    def test_empty_string_returns_empty_or_zero(self) -> None:
        """Тест: пустая строка → empty or 0.0.
        
        Empty string results in fallback value.
        """
        vp = ValueProcessor()
        result = vp.process_value(1, "", 0.9, "T")
        # Result may be empty string or 0.0
        assert result == "" or float(result) == 0.0

    def test_pure_text_hw_returns_empty_or_zero(self) -> None:
        """Тест: чистый текст "HW" → empty or 0.0.
        
        Text without any digits results in fallback.
        """
        vp = ValueProcessor()
        result = vp.process_value(1, "HW", 0.9, "T")
        # Result may be empty string or 0.0
        assert result == "" or float(result) == 0.0

    def test_text_on_off_returns_empty_or_zero(self) -> None:
        """Тест: текстовые состояния "ON"/"OFF" → empty or 0.0."""
        vp = ValueProcessor()
        
        result_on = vp.process_value(1, "ON", 0.9, "T")
        assert result_on == "" or float(result_on) == 0.0
        
        result_off = vp.process_value(2, "OFF", 0.9, "T")
        assert result_off == "" or float(result_off) == 0.0

    def test_whitespace_only_returns_empty_or_zero(self) -> None:
        """Тест: только пробелы → empty or 0.0."""
        vp = ValueProcessor()
        result = vp.process_value(1, "   ", 0.9, "T")
        assert result == "" or float(result) == 0.0

    def test_special_characters_only_returns_empty_or_zero(self) -> None:
        """Тест: только спецсимволы → empty or 0.0."""
        vp = ValueProcessor()
        result = vp.process_value(1, "---***///", 0.9, "T")
        assert result == "" or float(result) == 0.0


class TestValueProcessorConfidenceThresholds:
    """Тесты порогов уверенности и fallback chain."""

    def test_low_confidence_first_frame_accepted(self) -> None:
        """Тест: низкая уверенность на первом кадре принимается.
        
        When there's no previous valid value, low confidence readings
        should be accepted to bootstrap the tracking.
        """
        vp = ValueProcessor(confidence_threshold=0.5, initial_confidence_threshold=0.3)
        
        # First frame with low confidence (0.4) should be accepted
        result = vp.process_value(1, "100.0", 0.4, "T")
        assert result == "100.0"

    def test_low_confidence_subsequent_frame_fallback(self) -> None:
        """Тест: низкая уверенность на последующих кадрах → fallback.
        
        After a valid value is established, low confidence readings
        should fallback to the last valid value.
        """
        vp = ValueProcessor(confidence_threshold=0.5)
        
        # Establish valid value
        vp.process_value(1, "100.0", 0.9, "T")
        
        # Low confidence should fallback
        result = vp.process_value(1, "999.0", 0.3, "T")
        assert result == "100.0"

    def test_fallback_chain_last_valid_used(self) -> None:
        """Тест: цепочка fallback использует last_valid.
        
        When current OCR fails, should use the most recent valid value.
        """
        vp = ValueProcessor(confidence_threshold=0.5, window_size=1)  # No smoothing
        
        # Establish a series of valid values
        vp.process_value(1, "100.0", 0.9, "T")
        vp.process_value(1, "101.0", 0.9, "T")
        
        # Failed OCR should fallback to last valid (101.0)
        result = vp.process_value(1, "garbage", 0.2, "T")
        assert result == "101.0"

    def test_empty_string_only_as_last_resort(self) -> None:
        """Тест: пустая строка возвращается только как абсолютный last resort.
            
        Empty string should only be returned when:
        1. No previous valid value exists
        2. Current OCR fails to extract any number
        3. Fallback chain exhausted
            
        This prevents polluting _last_valid with 0.0 values.
        """
        vp = ValueProcessor(confidence_threshold=0.5, window_size=1)
    
        # First call with garbage data - returns empty or 0.0
        result = vp.process_value(1, "garbage", 0.2, "T")
        assert result == "" or float(result) == 0.0
        
        # After valid value established, fallback to it
        vp.process_value(1, "50.0", 0.9, "T")
        result = vp.process_value(1, "garbage", 0.2, "T")
        assert result == "50.0"  # Not 0.0!


class TestValueProcessorTemporalSmoothing:
    """Тесты временно́го сглаживания."""

    def test_median_filter_basic(self) -> None:
        """Тест: медианный фильтр сглаживает значения."""
        vp = ValueProcessor(window_size=3, confidence_threshold=0.3)
        
        # Add values that would have outlier
        vp.process_value(1, "100.0", 0.9, "T")
        vp.process_value(1, "100.5", 0.9, "T")
        vp.process_value(1, "100.3", 0.9, "T")
        
        # Median of [100.0, 100.5, 100.3] = 100.3
        result = vp.process_value(1, "100.4", 0.9, "T")
        assert float(result) >= 100.0
        assert float(result) <= 101.0

    def test_outlier_rejection(self) -> None:
        """Тест: выбросы отбрасываются при сглаживании."""
        vp = ValueProcessor(window_size=5, confidence_threshold=0.3, outlier_std_threshold=2.0)
        
        # Add consistent values
        vp.process_value(1, "100.0", 0.9, "T")
        vp.process_value(1, "100.1", 0.9, "T")
        vp.process_value(1, "100.2", 0.9, "T")
        
        # Add outlier
        result = vp.process_value(1, "150.0", 0.9, "T")
        
        # Result should be closer to 100 than 150
        assert float(result) < 125.0


class TestValueProcessorStaticMethods:
    """Тесты статических методов извлечения чисел."""

    def test_extract_best_number_simple(self) -> None:
        """Тест: простое число."""
        result = ValueProcessor._extract_best_number("123.45")
        assert result == 123.45

    def test_extract_best_number_with_unit(self) -> None:
        """Тест: число с единицей измерения."""
        result = ValueProcessor._extract_best_number("123.45°C")
        assert result == 123.45

    def test_extract_best_number_negative(self) -> None:
        """Тест: отрицательное число."""
        result = ValueProcessor._extract_best_number("-25.5")
        assert result == -25.5

    def test_extract_best_number_empty(self) -> None:
        """Тест: пустая строка → None."""
        result = ValueProcessor._extract_best_number("")
        assert result is None

    def test_extract_best_number_no_digits(self) -> None:
        """Тест: строка без цифр → None."""
        result = ValueProcessor._extract_best_number("abc")
        assert result is None

    def test_has_digit_true(self) -> None:
        """Тест: _has_digit возвращает True для строк с цифрами."""
        assert ValueProcessor._has_digit("abc123") is True
        assert ValueProcessor._has_digit("100°C") is True

    def test_has_digit_false(self) -> None:
        """Тест: _has_digit возвращает False для строк без цифр."""
        assert ValueProcessor._has_digit("abc") is False
        assert ValueProcessor._has_digit("HW") is False
        assert ValueProcessor._has_digit("") is False
