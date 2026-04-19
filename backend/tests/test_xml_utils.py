"""Тесты для XML-утилит."""

from app.utils.xml_utils import format_timestamp, validate_param_value


def test_format_timestamp() -> None:
    assert format_timestamp(0.5) == "00:00:00.500"
    assert format_timestamp(60.5) == "00:01:00.500"
    assert format_timestamp(3661.123) == "01:01:01.123"


def test_format_timestamp_zero() -> None:
    assert format_timestamp(0.0) == "00:00:00.000"


def test_validate_param_value_numeric() -> None:
    assert validate_param_value("12.5", decimal_places=1) is True
    assert validate_param_value("-3.14", decimal_places=2) is True
    assert validate_param_value("0", decimal_places=0) is True
    assert validate_param_value("100.0", decimal_places=1) is True


def test_validate_param_value_invalid() -> None:
    assert validate_param_value("abc", decimal_places=1) is False
    assert validate_param_value("", decimal_places=1) is False
    assert validate_param_value("12.5.3", decimal_places=1) is False


def test_validate_param_value_special() -> None:
    # Some params might have "N/A" or similar
    assert validate_param_value("N/A", decimal_places=1) is False
