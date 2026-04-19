"""Классификатор текстовых блоков: label, value, mixed, unknown.

Работает на regex + позиционных эвристиках — без обучения.
Адаптирован для SCADA/HMI мнемосхем: понимает единицы измерения,
шаблоны sensor ID (TI-101, P-205), inline-пары («Давление: 12.3 МПа»).
"""

from __future__ import annotations

import re

from app.core.ocr_models import BBox, BoxType, TextBox


# ---------------------------------------------------------------------------
# Паттерны для SCADA/HMI
# ---------------------------------------------------------------------------

# Единицы измерения на мнемосхемах
UNIT_STR = (
    r"МПа|кПа|Па|бар|bar|°C|℃|К|кВт|МВт|Вт|А|кА|В|кВ|"
    r"м³/ч|л/мин|об/мин|rpm|Гц|Hz|%|кг|т|мм|м|с|мин|ч|мс|"
    r"мм/с|м/с|т/ч|кг/ч|м³|л|мм рт\.ст"
)

UNIT_PATTERNS = re.compile(r"\b(" + UNIT_STR + r")\b", re.IGNORECASE)

# Числовое значение с опциональной единицей
VALUE_PATTERN = re.compile(
    r"^[-+]?\d[\d\s]*[.,]?\d*\s*(?:" + UNIT_STR + r")?$",
    re.IGNORECASE,
)

# Inline «Метка: значение» или «Метка = значение»
INLINE_PATTERN = re.compile(
    r"^(.+?)\s*[:=]\s*([-+]?\d[\d.,]*\s*\S*)$"
)

# Sensor ID: TI-101, P-205, dP-310, TE4401, FIC-502
# Также ловим OCR-ошибки: 11-101, T1-101 (когда буква распознана как цифра)
# Добавлены кириллические символы, похожие на латинские: Т→T, Р→P, и т.д.
SENSOR_ID_PATTERN = re.compile(
    r"^[A-Za-zА-Яа-я0-9]{1,4}[-]?\d{2,5}$"
)

# Строгий Sensor ID (с хотя бы одной буквой — латинской или кириллической)
SENSOR_ID_STRICT = re.compile(
    r"^[A-Za-zА-Яа-я]{1,4}[-]?\d{2,5}$"
)

# Чисто буквенная метка (русский/английский + спецсимволы)
LABEL_PATTERN = re.compile(
    r"^[А-Яа-яA-Za-z\s\(\)\/\-\.№#]+$"
)


class TextClassifier:
    """Классифицирует TextBox как label, value, mixed или unknown.

    Приоритет классификации:
    1. По содержимому (regex)
    2. По позиции (если контент неоднозначен)
    """

    def classify(self, box: TextBox, all_boxes: list[TextBox]) -> TextBox:
        """Классифицирует один TextBox с учётом контекста."""
        box.box_type = self._classify_by_content(box.text)
        if box.box_type == "unknown":
            box.box_type = self._classify_by_position(box, all_boxes)
        return box

    def classify_all(self, boxes: list[TextBox]) -> list[TextBox]:
        """Классифицирует все TextBox-ы."""
        return [self.classify(b, boxes) for b in boxes]

    def split_inline(self, box: TextBox) -> tuple[TextBox, TextBox] | None:
        """Разбивает mixed-box («Давление: 12.3 МПа») на два TextBox."""
        m = INLINE_PATTERN.match(box.text.strip())
        if not m:
            return None
        label_text, value_text = m.group(1), m.group(2)
        # Делим bbox пропорционально длине строк
        ratio = len(label_text) / len(box.text)
        b = box.bbox
        label_w = int(b.w * ratio)

        label_box = TextBox(
            bbox=BBox(b.x, b.y, label_w, b.h),
            text=label_text.strip(),
            confidence=box.confidence,
            source=box.source,
            box_type="label",
        )
        value_box = TextBox(
            bbox=BBox(b.x + label_w, b.y, b.w - label_w, b.h),
            text=value_text.strip(),
            confidence=box.confidence,
            source=box.source,
            box_type="value",
        )
        return label_box, value_box

    # -----------------------------------------------------------------------
    # Приватные методы
    # -----------------------------------------------------------------------

    def _classify_by_content(self, text: str) -> BoxType:
        """Классификация по содержимому текста."""
        text = text.strip()
        if not text:
            return "unknown"

        # Inline пара «Метка: значение»
        if INLINE_PATTERN.match(text):
            return "mixed"

        # Sensor ID: TI-101, P-205, или OCR-ошибка 11-101
        if SENSOR_ID_PATTERN.match(text):
            # Если есть хотя бы одна буква — точно метка
            # Если только цифры + дефис — скорее всего метка с OCR-ошибкой
            if SENSOR_ID_STRICT.match(text):
                return "label"
            # Цифры + дефис: проверяем структуру NNN-NNN (sensor ID pattern)
            if re.match(r"^\d{1,2}-\d{2,5}$", text.strip()):
                return "label"

        # Чисто числовое значение (758.3, -12.5, 0.25)
        cleaned = text.replace(",", ".").strip()
        if re.match(r"^-?\d+\.?\d*$", cleaned):
            return "value"

        # Число + единица измерения → value
        if VALUE_PATTERN.match(text):
            return "value"

        # Содержит цифры + единицы → скорее всего value
        if re.search(r"\d", text) and UNIT_PATTERNS.search(text):
            return "value"

        # Только буквы/символы → метка
        if LABEL_PATTERN.match(text):
            return "label"

        # Кириллица → метка (названия параметров на русском)
        if re.search(r"[а-яА-Я]", text):
            return "label"

        # Смешанный текст с цифрами — проверяем на inline
        if re.search(r"\d", text):
            # Если число в конце — скорее всего value
            if re.search(r"\d+\.?\d*\s*$", text):
                return "value"

        return "unknown"

    def _classify_by_position(
        self, box: TextBox, all_boxes: list[TextBox]
    ) -> BoxType:
        """Позиционная классификация для неоднозначных случаев.

        В HMI-экранах:
        - Левые колонки — обычно метки
        - Правые колонки — обычно значения
        - Верхняя позиция в паре — метка, нижняя — значение
        """
        if not all_boxes:
            return "unknown"

        cx = box.bbox.center[0]
        all_cx = [b.bbox.center[0] for b in all_boxes if b.text]
        if not all_cx:
            return "unknown"

        median_cx = sorted(all_cx)[len(all_cx) // 2]

        # Значительно левее медианы → метка
        if cx < median_cx * 0.8:
            return "label"
        # Значительно правее медианы → значение
        if cx > median_cx * 1.2:
            return "value"

        return "unknown"
