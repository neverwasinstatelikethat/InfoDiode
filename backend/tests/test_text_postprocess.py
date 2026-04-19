"""Тесты постобработки текста."""

from app.utils.text_postprocess import (
    extract_numeric,
    format_decimal,
    substitute_chars,
    validate_change_rate,
    validate_range,
)


class TestTextPostprocess:
    """Тесты постобработки распознанного текста."""

    def test_substitute_chars(self) -> None:
        """Тест подстановки символов."""
        assert substitute_chars("O") == "0"
        assert substitute_chars("l") == "1"
        assert substitute_chars("S") == "5"
        assert substitute_chars("B") == "8"
        assert substitute_chars(",") == "."
        assert substitute_chars("7S8,3") == "758.3"

    def test_extract_numeric(self) -> None:
        """Тест извлечения чисел."""
        assert extract_numeric("758.3") == 758.3
        assert extract_numeric("758,3") == 758.3
        assert extract_numeric("-12.5") == -12.5
        assert extract_numeric("T=45.2 C") == 45.2
        assert extract_numeric("abc") is None

    def test_validate_range(self) -> None:
        """Тест валидации диапазона."""
        assert validate_range(100.0, "T") is True
        assert validate_range(-25.0, "T") is True  # в пределах [-50, 800]
        assert validate_range(-60.0, "T") is False  # ниже -50
        assert validate_range(-100.0, "T") is False  # ниже -50
        assert validate_range(900.0, "T") is False  # выше 800
        assert validate_range(50.0, "Vb") is True  # граничное значение включено
        assert validate_range(50.1, "Vb") is False  # выше максимума

    def test_validate_change_rate(self) -> None:
        """Тест валидации скорости изменения."""
        # Температура: max 5°C/s, интервал 0.5s -> max 2.5°C за шаг
        assert validate_change_rate(100.0, 102.0, "T", 0.5) is True
        assert validate_change_rate(100.0, 105.0, "T", 0.5) is False  # +5°C за 0.5s = 10°C/s

    def test_format_decimal(self) -> None:
        """Тест форматирования десятичных знаков."""
        assert format_decimal(758.3, "T") == "758.3"     # 1 dp
        assert format_decimal(4.21, "P") == "4.21"       # 2 dp
        assert format_decimal(45.0, "n") == "45"         # 0 dp
        assert format_decimal(0.25, "Vb") == "0.25"      # 2 dp
