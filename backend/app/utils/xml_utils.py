"""Утилиты для работы с XML."""

from __future__ import annotations

import re


def format_timestamp(seconds: float) -> str:
    """Форматирует секунды в таймстемп HH:MM:SS.mmm.

    Args:
        seconds: Время в секундах.

    Returns:
        Строка в формате HH:MM:SS.mmm.
    """
    total_ms = int(seconds * 1000)
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def validate_timestamp_format(timestamp: str) -> bool:
    """Проверяет формат таймстемпа HH:MM:SS.mmm.

    Args:
        timestamp: Строка таймстемпа.

    Returns:
        True если формат корректный.
    """
    pattern = r"^\d{2}:\d{2}:\d{2}\.\d{3}$"
    return bool(re.match(pattern, timestamp))


def validate_param_value(value: str, decimal_places: int) -> bool:
    """Проверяет формат значения параметра.

    Args:
        value: Строковое значение.
        decimal_places: Ожидаемое количество знаков после запятой.

    Returns:
        True если формат корректный.
    """
    if decimal_places == 0:
        return bool(re.match(r"^-?\d+$", value))

    pattern = rf"^-?\d+\.\d{{{decimal_places}}}$"
    return bool(re.match(pattern, value))
