"""Column-aware Layout Analyzer для SCADA мнемосхем.

SCADA-экраны имеют колоночную структуру: ID | Value | Unit.
Алгоритм использует X-кластеризацию для обнаружения колонок,
затем сопоставляет строки по Y-близости с компенсацией наклона.

Алгоритм:
1. Адаптивная X-кластеризация: bandwidth вычисляется по MAD зазоров
   (робастнее фиксированного фактора ширины кадра)
2. Колонки сортируются слева направо
3. Валидация колонок: левая = label, средняя = value, правая = unit (опционально)
4. Y-сопоставление с компенсацией наклона камеры:
   ранговое сопоставление вместо абсолютного Y-допуска
5. Улучшенная функция уверенности: учитывает X-выравнивание,
   содержимое текста и char-уверенность
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Protocol

from app.core.ocr_models import BBox, BoxType, TextBox, TextPair
from app.core.text_classifier import TextClassifier, INLINE_PATTERN


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayoutConfig:
    """Конфигурация ColumnLayoutAnalyzer.

    Attributes:
        bandwidth_factor: Доля ширины кадра для начальной оценки bandwidth.
        y_tolerance_factor: Множитель медианной высоты для Y-допуска.
        min_column_gap_factor: Минимальный зазор между колонками (× bandwidth).
        tilt_compensation: Включить компенсацию наклона камеры (ранговое сопоставление).
        min_column_purity: Минимальная доля ожидаемого типа в колонке для валидации.
        enable_unit_column: Искать 3-ю колонку (единицы измерения).
        rank_matching: Использовать ранговое сопоставление вместо абсолютного Y.
    """

    bandwidth_factor: float = 0.05
    y_tolerance_factor: float = 0.6
    min_column_gap_factor: float = 0.5
    tilt_compensation: bool = True
    min_column_purity: float = 0.3
    enable_unit_column: bool = True
    rank_matching: bool = True


# ---------------------------------------------------------------------------
# Протокол для возможной замены на ML-модель в будущем
# ---------------------------------------------------------------------------

class LayoutAnalyzer(Protocol):
    """Протокол анализатора layout-а.

    Позволяет подставить ML-модель (LayoutLMv3 и т.д.) вместо rule-based,
    не меняя остальной код.
    """

    def extract_pairs(self, boxes: list[TextBox]) -> list[TextPair]: ...


# ---------------------------------------------------------------------------
# Реализация
# ---------------------------------------------------------------------------

class ColumnLayoutAnalyzer:
    """Анализатор колоночной структуры SCADA-экранов.

    Использует адаптивную X-кластеризацию для обнаружения колонок,
    валидацию типов колонок и Y-сопоставление с компенсацией наклона
    для связывания меток со значениями в пределах одной строки.
    """

    def __init__(self, config: LayoutConfig | None = None) -> None:
        """Инициализирует анализатор.

        Args:
            config: Конфигурация анализатора. Если None — используются дефолты.
        """
        self._cfg = config or LayoutConfig()
        self._classifier = TextClassifier()

    def extract_pairs(self, boxes: list[TextBox]) -> list[TextPair]:
        """Извлекает пары label:value на основе колоночного анализа.

        Args:
            boxes: Список распознанных текстовых блоков (уже классифицированных).

        Returns:
            Список пар метка-значение.
        """
        if len(boxes) < 2:
            return []

        # 1. Кластеризуем по X (обнаруживаем колонки)
        columns = self._cluster_columns(boxes)

        if len(columns) < 2:
            # Одна колонка — пробуем inline-разбиение
            return self._extract_inline_pairs(boxes)

        # 2. Сортируем колонки слева направо
        columns.sort(key=lambda c: statistics.mean(b.bbox.center[0] for b in c))

        # 3. Валидируем колонки (left=label, right=value, far-right=unit)
        columns = self._validate_columns(columns)

        if len(columns) < 2:
            return self._extract_inline_pairs(boxes)

        # 4. Сопоставляем строки между колонками
        if self._cfg.rank_matching and self._cfg.tilt_compensation:
            return self._match_rows_ranked(columns)
        return self._match_rows(columns)

    # -----------------------------------------------------------------------
    # Адаптивная X-кластеризация
    # -----------------------------------------------------------------------

    def _cluster_columns(
        self,
        boxes: list[TextBox],
    ) -> list[list[TextBox]]:
        """X-кластеризация через адаптивную gap-based группировку.

        Алгоритм:
        1. Сортируем боксы по X-центру
        2. Вычисляем все зазоры между соседями
        3. Определяем порог разделения через MAD (Median Absolute Deviation)
           вместо фиксированного bandwidth — робастнее к неравномерным layout-ам
        4. Разрыв > threshold → новая колонка

        Args:
            boxes: Список текстовых блоков.

        Returns:
            Список колонок (каждая колонка — список TextBox).
        """
        if not boxes:
            return []

        # Сортируем по X-центру
        sorted_boxes = sorted(boxes, key=lambda b: b.bbox.center[0])
        x_centers = [b.bbox.center[0] for b in sorted_boxes]

        # Вычисляем все зазоры
        gaps = [x_centers[i] - x_centers[i - 1] for i in range(1, len(x_centers))]

        if not gaps:
            return [sorted_boxes]

        # Адаптивный порог: медиана + k * MAD
        # MAD робастен к выбросам (не смещается большими межколонными зазорами)
        threshold = self._adaptive_threshold(gaps, sorted_boxes)

        # Gap-based clustering
        columns: list[list[TextBox]] = []
        current_column: list[TextBox] = [sorted_boxes[0]]

        for i in range(1, len(sorted_boxes)):
            gap = x_centers[i] - x_centers[i - 1]

            if gap > threshold:
                columns.append(current_column)
                current_column = [sorted_boxes[i]]
            else:
                current_column.append(sorted_boxes[i])

        if current_column:
            columns.append(current_column)

        return columns

    def _adaptive_threshold(self, gaps: list[float], boxes: list[TextBox]) -> float:
        """Вычисляет адаптивный порог разделения колонок.

        Использует MAD (Median Absolute Deviation) зазоров — робастный
        к выбросам метод. При малом числе зазоров (<=3) MAD ненадёжен,
        и fallback на bandwidth_factor × ширина_кадра.

        Args:
            gaps: Зазоры между X-центрами соседних боксов.
            boxes: Исходные боксы (для вычисления ширины кадра).

        Returns:
            Порог разделения колонок в пикселях.
        """
        # Классический bandwidth — всегда доступен как fallback
        frame_w = max(b.bbox.x + b.bbox.w for b in boxes)
        bandwidth = max(frame_w * self._cfg.bandwidth_factor, 30.0)

        # При малом числе зазоров MAD ненадёжен → используем bandwidth
        if len(gaps) <= 3:
            return bandwidth

        median_gap = statistics.median(gaps)
        # MAD = медиана |xi - медиана|
        mad = statistics.median(abs(g - median_gap) for g in gaps)

        # Порог: медиана + k × MAD (k ≈ 2.5)
        # При нулевом MAD (все зазоры одинаковые) — fallback
        if mad < 1e-6:
            return bandwidth

        threshold = median_gap + 2.5 * mad

        # Берём минимум из адаптивного и классического — адаптивный
        # более точен при достаточном числе зазоров, но не должен
        # быть слишком большим (иначе одна колонка не разобьётся)
        return min(threshold, max(bandwidth, median_gap * 1.5))

    # -----------------------------------------------------------------------
    # Валидация колонок
    # -----------------------------------------------------------------------

    def _validate_columns(
        self,
        columns: list[list[TextBox]],
    ) -> list[list[TextBox]]:
        """Валидирует колонки по содержимому.

        Проверяет, что левая колонка содержит labels, правая — values.
        Если 3+ колонок и включён enable_unit_column — проверяет правую
        на наличие единиц измерения.

        Args:
            columns: Список колонок, отсортированных слева направо.

        Returns:
            Валидированный список колонок (может быть сокращён).
        """
        if len(columns) < 2:
            return columns

        # Анализируем типичный тип содержимого в каждой колонке
        col_types = []
        for col in columns:
            types = [b.box_type for b in col if b.text.strip()]
            if not types:
                col_types.append("unknown")
                continue
            # Считаем доминирующий тип
            label_ratio = sum(1 for t in types if t in ("label", "mixed")) / len(types)
            value_ratio = sum(1 for t in types if t in ("value",)) / len(types)
            if label_ratio >= self._cfg.min_column_purity:
                col_types.append("label")
            elif value_ratio >= self._cfg.min_column_purity:
                col_types.append("value")
            else:
                col_types.append("unknown")

        # Если левая колонка — value, а правая — label, возможно перевёрнутый layout
        if col_types[0] == "value" and col_types[-1] == "label":
            columns.reverse()
            col_types.reverse()

        # Если включён 3-й столбец (units), сохраняем его
        if self._cfg.enable_unit_column and len(columns) >= 3:
            # Проверяем, что 3-я колонка содержит единицы
            unit_col = columns[2]
            unit_ratio = self._compute_unit_ratio(unit_col)
            if unit_ratio >= self._cfg.min_column_purity:
                return columns[:3]
            # 3-я колонка не похожа на units — возможно 2-колоночный layout с шумом
            return columns[:2]

        return columns[:2] if len(columns) > 2 else columns

    def _compute_unit_ratio(self, boxes: list[TextBox]) -> float:
        """Вычисляет долю боксов, похожих на единицы измерения.

        Args:
            boxes: Список текстовых блоков в колонке.

        Returns:
            Доля [0, 1] боксов, содержащих единицы измерения.
        """
        if not boxes:
            return 0.0
        unit_count = 0
        for b in boxes:
            text = b.text.strip()
            if not text:
                continue
            # Короткие строки с буквами/символами, без цифр в начале
            if len(text) <= 8 and not text[0].isdigit():
                # Проверяем на известные единицы
                if _looks_like_unit(text):
                    unit_count += 1
        return unit_count / max(len([b for b in boxes if b.text.strip()]), 1)

    # -----------------------------------------------------------------------
    # Y-сопоставление: ранговое (с компенсацией наклона)
    # -----------------------------------------------------------------------

    def _match_rows_ranked(
        self,
        columns: list[list[TextBox]],
    ) -> list[TextPair]:
        """Ранговое Y-сопоставление с компенсацией наклона камеры.

        Вместо абсолютного Y-допуска используем ранговое сопоставление:
        сортируем каждую колонку по Y, затем сопоставляем по порядку
        (1-й label → 1-й value, 2-й → 2-й и т.д.).

        Это компенсирует наклон камеры: даже если Y-координаты
        смещены, порядок элементов в колонке сохраняется.

        Args:
            columns: Список колонок (каждая — список TextBox), отсортирован слева направо.

        Returns:
            Список пар метка-значение.
        """
        if len(columns) < 2:
            return []

        label_col = columns[0]
        value_col = columns[1]

        # Сортируем каждую колонку по Y
        label_sorted = sorted(
            [b for b in label_col if b.box_type in ("label", "mixed", "unknown")],
            key=lambda b: b.bbox.center[1],
        )
        value_sorted = sorted(
            [b for b in value_col if b.box_type in ("value", "unknown")],
            key=lambda b: b.bbox.center[1],
        )

        # Адаптивный Y-допуск для проверки правдоподобия пары
        all_heights = [b.bbox.h for col in columns for b in col]
        median_height = statistics.median(all_heights) if all_heights else 24.0
        y_tolerance = median_height * self._cfg.y_tolerance_factor * 2  # удвоенный для рангового

        pairs: list[TextPair] = []
        used_values: set[int] = set()

        # Ранговое сопоставление: для каждого label ищем value с ближайшим рангом
        for rank, label_box in enumerate(label_sorted):
            # Ожидаемый ранок value = rank (тот же порядок)
            # Ищем в окне [rank-1, rank+1] для устойчивости к пропускам
            search_start = max(0, rank - 1)
            search_end = min(len(value_sorted), rank + 2)

            best_value: TextBox | None = None
            best_dy = float("inf")

            for vi in range(search_start, search_end):
                value_box = value_sorted[vi]
                if id(value_box) in used_values:
                    continue

                dy = abs(value_box.bbox.center[1] - label_box.bbox.center[1])

                # Проверяем правдоподобие Y-расстояния
                if dy <= y_tolerance and dy < best_dy:
                    best_value = value_box
                    best_dy = dy

            if best_value is not None:
                used_values.add(id(best_value))

                pair_confidence = self._compute_pair_confidence(
                    label_box, best_value, best_dy, y_tolerance,
                )

                # Определяем unit, если есть 3-я колонка
                unit_box = None
                if len(columns) >= 3 and self._cfg.enable_unit_column:
                    unit_box = self._find_unit(columns[2], best_value, y_tolerance)

                pairs.append(
                    TextPair(
                        label=label_box,
                        value=best_value,
                        relation="horizontal",
                        pair_confidence=pair_confidence,
                    )
                )

        return pairs

    # -----------------------------------------------------------------------
    # Y-сопоставление: классическое (абсолютный Y-допуск)
    # -----------------------------------------------------------------------

    def _match_rows(
        self,
        columns: list[list[TextBox]],
    ) -> list[TextPair]:
        """Сопоставляет строки между соседними колонками по Y-близости.

        Первые две колонки = label/value. Третья колонка = units (опционально).

        Args:
            columns: Список колонок (каждая — список TextBox), отсортирован слева направо.

        Returns:
            Список пар метка-значение.
        """
        if len(columns) < 2:
            return []

        # Вычисляем адаптивный Y-допуск
        all_heights = [b.bbox.h for col in columns for b in col]
        median_height = statistics.median(all_heights) if all_heights else 24.0
        y_tolerance = median_height * self._cfg.y_tolerance_factor

        # Первая колонка = labels, вторая = values
        label_col = columns[0]
        value_col = columns[1]

        # Сортируем каждую колонку по Y
        label_col_sorted = sorted(label_col, key=lambda b: b.bbox.center[1])
        value_col_sorted = sorted(value_col, key=lambda b: b.bbox.center[1])

        pairs: list[TextPair] = []
        used_values: set[int] = set()

        for label_box in label_col_sorted:
            # Проверяем, что это действительно label
            if label_box.box_type not in ("label", "unknown"):
                continue

            label_y = label_box.bbox.center[1]
            best_value: TextBox | None = None
            best_dy = float("inf")

            for value_box in value_col_sorted:
                # Пропускаем уже использованные values
                if id(value_box) in used_values:
                    continue

                # Проверяем, что это value
                if value_box.box_type not in ("value", "unknown"):
                    continue

                value_y = value_box.bbox.center[1]
                dy = abs(value_y - label_y)

                # Y-близость в пределах допуска
                if dy <= y_tolerance and dy < best_dy:
                    best_value = value_box
                    best_dy = dy

            if best_value is not None:
                used_values.add(id(best_value))

                pair_confidence = self._compute_pair_confidence(
                    label_box, best_value, best_dy, y_tolerance,
                )

                # Определяем unit, если есть 3-я колонка
                unit_box = None
                if len(columns) >= 3 and self._cfg.enable_unit_column:
                    unit_box = self._find_unit(columns[2], best_value, y_tolerance)

                pairs.append(
                    TextPair(
                        label=label_box,
                        value=best_value,
                        relation="horizontal",
                        pair_confidence=pair_confidence,
                    )
                )

        return pairs

    # -----------------------------------------------------------------------
    # Поиск единицы измерения
    # -----------------------------------------------------------------------

    def _find_unit(
        self,
        unit_col: list[TextBox],
        value_box: TextBox,
        y_tolerance: float,
    ) -> TextBox | None:
        """Ищет единицу измерения в 3-й колонке для данной value.

        Args:
            unit_col: Колонка единиц измерения.
            value_box: Значение, для которого ищем единицу.
            y_tolerance: Y-допуск для сопоставления.

        Returns:
            TextBox с единицей измерения или None.
        """
        value_y = value_box.bbox.center[1]
        best_unit: TextBox | None = None
        best_dy = float("inf")

        for unit_box in unit_col:
            dy = abs(unit_box.bbox.center[1] - value_y)
            if dy <= y_tolerance and dy < best_dy:
                if _looks_like_unit(unit_box.text.strip()):
                    best_unit = unit_box
                    best_dy = dy

        return best_unit

    # -----------------------------------------------------------------------
    # Улучшенная функция уверенности
    # -----------------------------------------------------------------------

    def _compute_pair_confidence(
        self,
        label: TextBox,
        value: TextBox,
        dy: float,
        y_tolerance: float,
    ) -> float:
        """Вычисляет уверенность в паре label:value.

        Учитывает:
        - Произведение OCR-уверенностей (базовый фактор)
        - Y-расстояние (штраф за удалённость)
        - X-выравнивание (бонус за правильный порядок колонок)
        - Валидацию содержимого (value должен быть числовым)

        Args:
            label: Текстовый блок метки.
            value: Текстовый блок значения.
            dy: Разница Y-координат.
            y_tolerance: Y-допуск.

        Returns:
            Уверенность в диапазоне [0, 1].
        """
        # Базовая уверенность = произведение уверенности OCR
        base_conf = label.confidence * value.confidence

        # Штраф за Y-расстояние
        y_penalty = dy / (y_tolerance * 2 + 1e-6)

        # Бонус за X-выравнивание: label левее value = правильно
        x_bonus = 0.05 if label.bbox.center[0] < value.bbox.center[0] else -0.1

        # Бонус за числовое содержимое value
        content_bonus = 0.05 if _looks_like_value(value.text) else -0.05

        # Используем char_confs если доступны (более точная оценка)
        if value.char_confs:
            char_conf_avg = statistics.mean(value.char_confs)
            # Взвешиваем: 70% char_confs + 30% box confidence
            refined_value_conf = 0.7 * char_conf_avg + 0.3 * value.confidence
            base_conf = label.confidence * refined_value_conf

        # Итоговая уверенность
        pair_conf = base_conf + x_bonus + content_bonus - y_penalty

        return max(0.0, min(1.0, pair_conf))

    # -----------------------------------------------------------------------
    # Inline fallback
    # -----------------------------------------------------------------------

    def _extract_inline_pairs(self, boxes: list[TextBox]) -> list[TextPair]:
        """Fallback для одноколоночного layout: поиск inline-пар.

        Ищет паттерны «Label: Value» в пределах одного TextBox.

        Args:
            boxes: Список текстовых блоков.

        Returns:
            Список пар, извлечённых из inline-паттернов.
        """
        pairs: list[TextPair] = []

        for box in boxes:
            # Пробуем разбить inline-пару
            split = self._classifier.split_inline(box)
            if split is None:
                continue

            label_box, value_box = split

            # Уверенность = минимум из двух (с учётом возможной ошибки разбиения)
            pair_confidence = min(label_box.confidence, value_box.confidence) * 0.9

            pairs.append(
                TextPair(
                    label=label_box,
                    value=value_box,
                    relation="inline",
                    pair_confidence=max(0.0, min(1.0, pair_confidence)),
                )
            )

        return pairs


# ---------------------------------------------------------------------------
# Вспомогательные функции (module-level для тестируемости)
# ---------------------------------------------------------------------------

# Кэши скомпилированных паттернов (компилируются один раз)
_UNIT_RE = None
_VALUE_RE = None


def _get_unit_pattern():
    """Ленивая компиляция паттерна единиц измерения."""
    global _UNIT_RE
    if _UNIT_RE is None:
        import re
        UNIT_STR = (
            r"МПа|кПа|Па|бар|bar|°C|℃|К|кВт|МВт|Вт|А|кА|В|кВ|"
            r"м³/ч|л/мин|об/мин|rpm|Гц|Hz|%|кг|т|мм|м|с|мин|ч|мс|"
            r"мм/с|м/с|т/ч|кг/ч|м³|л|мм рт\.ст"
        )
        _UNIT_RE = re.compile(r"^\s*(" + UNIT_STR + r")\s*$", re.IGNORECASE)
    return _UNIT_RE


def _get_value_pattern():
    """Ленивая компиляция паттерна числового значения."""
    global _VALUE_RE
    if _VALUE_RE is None:
        import re
        _VALUE_RE = re.compile(r"^-?\d+[,.]?\d*$")
    return _VALUE_RE


def _looks_like_unit(text: str) -> bool:
    """Проверяет, похож ли текст на единицу измерения.

    Args:
        text: Текст для проверки.

    Returns:
        True если текст похож на единицу измерения.
    """
    if not text:
        return False
    return bool(_get_unit_pattern().match(text))


def _looks_like_value(text: str) -> bool:
    """Проверяет, похож ли текст на числовое значение.

    Args:
        text: Текст для проверки.

    Returns:
        True если текст похож на число.
    """
    if not text:
        return False
    cleaned = text.strip().replace(",", ".")
    return bool(_get_value_pattern().match(cleaned))
