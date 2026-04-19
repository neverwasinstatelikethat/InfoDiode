"""Постобработка распознанного текста.

Подстановка символов, исправление кириллицы,
извлечение чисел, валидация по диапазону.
"""

from __future__ import annotations

import re


# Таблица подстановки символов для числовых значений
# Включает латинские и кириллические символы, часто путаемые OCR
CHAR_SUBSTITUTIONS: dict[str, str] = {
    # Латинские
    "O": "0",
    "o": "0",
    "l": "1",
    "I": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    # Кириллические → цифры
    "О": "0",  # Cyrillic О → 0
    "о": "0",
    "З": "3",  # Cyrillic З → 3
    "з": "3",
    # Кириллические → латинские (в контексте чисел)
    "б": "6",  # Cyrillic б → 6
    "Ь": "b",  # Cyrillic Ь → b (редко)
    "ь": "b",
    "г": "r",  # Cyrillic г → r (редко)
    "п": "n",  # Cyrillic п → n (редко)
    # Разделители
    ",": ".",
    "—": "-",
    "−": "-",
    "–": "-",
}

# Максимальная скорость изменения по типам параметров (в секунду)
MAX_CHANGE_RATE: dict[str, float] = {
    "T": 5.0,       # °C/s
    "P": 50.0,      # кПа/s
    "dP": 10.0,     # кПа/s
    "Vb": 5.0,      # мм/с²/s
    "L": 10.0,      # мм/s или %/s
    "n": 10.0,      # Гц/s
    "Pos": 5.0,     # %/s
    "V": 10.0,      # В/s
    "f": 2.0,       # %/s
    "R": 0.0,       # Резерв — без ограничения
}

# Диапазоны допустимых значений по типам
VALUE_RANGES: dict[str, tuple[float, float]] = {
    "T": (-50.0, 800.0),
    "P": (0.0, 8000.0),
    "dP": (-100.0, 100.0),
    "Vb": (0.0, 50.0),
    "L": (0.0, 1000.0),
    "n": (0.0, 300.0),
    "Pos": (0.0, 100.0),
    "V": (0.0, 400.0),
    "f": (0.0, 100.0),
    "R": (0.0, 9999.0),
}

# Количество знаков после запятой по типам
DECIMAL_PLACES: dict[str, int] = {
    "T": 1,
    "P": 2,
    "dP": 1,
    "Vb": 2,
    "L": 1,
    "n": 0,
    "Pos": 1,
    "V": 1,
    "f": 1,
    "R": 0,
}


def substitute_chars(text: str) -> str:
    """Заменяет часто путаемые OCR-символы в числовых значениях.

    Args:
        text: Распознанный текст.

    Returns:
        Текст с исправленными символами.
    """
    result = []
    for char in text:
        result.append(CHAR_SUBSTITUTIONS.get(char, char))
    return "".join(result)


def _strip_units(text: str) -> str:
    """Удаляет единицы измерения из текста значения.

    Args:
        text: Текст, возможно содержащий единицы измерения.

    Returns:
        Текст без единиц измерения.
    """
    # Русские и латинские единицы измерения
    unit_pattern = r"\s*(°C|℃|°|кПа|Па|МПа|бар|bar|мм/с|мм|м|кг|т|%|Гц|Hz|В|Вт|кВт|МВт|А|ед\.?|units?)$"
    return re.sub(unit_pattern, "", text, flags=re.IGNORECASE)


def _normalize_decimal_separator(text: str) -> str:
    """Нормализует десятичный разделитель (запятая → точка).

    Обрабатывает русский формат чисел: 758,3 → 758.3

    Args:
        text: Текст с возможной запятой как разделителем.

    Returns:
        Текст с точкой как десятичным разделителем.
    """
    # Заменяем запятую между цифрами на точку
    return re.sub(r"(\d),(\d)", r"\1.\2", text)


def _remove_spaces_in_numbers(text: str) -> str:
    """Удаляет пробелы внутри чисел.

    Обрабатывает OCR-ошибки: 7 58.3 → 758.3

    Args:
        text: Текст с возможными пробелами в числах.

    Returns:
        Текст без пробелов внутри чисел.
    """
    # Удаляем пробелы между цифрами
    return re.sub(r"(\d)\s+(\d)", r"\1\2", text)


def _fix_multiple_dots(text: str) -> str:
    """Исправляет множественные точки в числах.

    Обрабатывает OCR-ошибки: 758..3 → 758.3

    Args:
        text: Текст с возможными множественными точками.

    Returns:
        Текст с исправленными точками.
    """
    # Заменяем множественные точки на одну
    return re.sub(r"\.{2,}", ".", text)


def extract_numeric(text: str) -> float | None:
    """Извлекает числовое значение из текста.

    Применяет множественные этапы очистки:
    1. Подстановка символов (O→0, З→3, и т.д.)
    2. Удаление единиц измерения
    3. Нормализация десятичного разделителя (, → .)
    4. Удаление пробелов внутри чисел
    5. Исправление множественных точек

    Args:
        text: Текст, возможно содержащий число.

    Returns:
        Числовое значение или None.
    """
    if not text:
        return None

    cleaned = text.strip()

    # Этап 1: Подстановка символов
    cleaned = substitute_chars(cleaned)

    # Этап 2: Удаление единиц измерения
    cleaned = _strip_units(cleaned)

    # Этап 3: Нормализация десятичного разделителя
    cleaned = _normalize_decimal_separator(cleaned)

    # Этап 4: Удаление пробелов внутри чисел
    cleaned = _remove_spaces_in_numbers(cleaned)

    # Этап 5: Исправление множественных точек
    cleaned = _fix_multiple_dots(cleaned)

    match = re.search(r"-?\d+\.?\d*", cleaned)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def validate_range(
    value: float,
    param_type: str,
) -> bool:
    """Проверяет, попадает ли значение в допустимый диапазон.

    Args:
        value: Числовое значение.
        param_type: Тип параметра (T, P, dP, и т.д.).

    Returns:
        True если значение в допустимом диапазоне.
    """
    if param_type not in VALUE_RANGES:
        return True  # Неизвестный тип — не ограничиваем

    min_val, max_val = VALUE_RANGES[param_type]
    return min_val <= value <= max_val


def validate_change_rate(
    old_value: float,
    new_value: float,
    param_type: str,
    interval_s: float = 0.5,
) -> bool:
    """Проверяет, допустима ли скорость изменения значения.

    Args:
        old_value: Предыдущее значение.
        new_value: Новое значение.
        param_type: Тип параметра.
        interval_s: Временной интервал в секундах.

    Returns:
        True если скорость изменения допустима.
    """
    if param_type not in MAX_CHANGE_RATE:
        return True

    max_rate = MAX_CHANGE_RATE[param_type]
    if max_rate == 0:
        return True  # Резерв

    actual_rate = abs(new_value - old_value) / interval_s
    return actual_rate <= max_rate


def format_decimal(value: float, param_type: str) -> str:
    """Форматирует значение с нужным количеством знаков после запятой.

    Args:
        value: Числовое значение.
        param_type: Тип параметра.

    Returns:
        Отформатированная строка.
    """
    places = DECIMAL_PLACES.get(param_type, 1)
    return f"{value:.{places}f}"
