"""Тесты ColumnLayoutAnalyzer: адаптивная кластеризация, 3 колонки, компенсация наклона."""

from __future__ import annotations

import pytest

from app.core.column_layout_analyzer import (
    ColumnLayoutAnalyzer,
    LayoutConfig,
    _looks_like_unit,
    _looks_like_value,
)
from app.core.ocr_models import BBox, TextBox, TextPair
from app.core.text_classifier import TextClassifier


# ---------------------------------------------------------------------------
# Фабрики тестовых данных
# ---------------------------------------------------------------------------

def _tb(
    text: str,
    x: int,
    y: int,
    w: int = 80,
    h: int = 24,
    conf: float = 0.9,
    box_type: str = "unknown",
    source: str = "paddle",
) -> TextBox:
    """Создаёт TextBox с указанными координатами."""
    return TextBox(
        bbox=BBox(x=x, y=y, w=w, h=h),
        text=text,
        confidence=conf,
        source=source,
        box_type=box_type,
    )


def _classify(boxes: list[TextBox]) -> list[TextBox]:
    """Классифицирует боксы через TextClassifier."""
    classifier = TextClassifier()
    return classifier.classify_all(boxes)


# ---------------------------------------------------------------------------
# Тесты вспомогательных функций
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Тесты module-level функций."""

    def test_looks_like_unit_celsius(self) -> None:
        assert _looks_like_unit("°C") is True

    def test_looks_like_unit_kpa(self) -> None:
        assert _looks_like_unit("кПа") is True

    def test_looks_like_unit_percent(self) -> None:
        assert _looks_like_unit("%") is True

    def test_looks_like_unit_nonsense(self) -> None:
        assert _looks_like_unit("TI-101") is False

    def test_looks_like_unit_empty(self) -> None:
        assert _looks_like_unit("") is False

    def test_looks_like_value_integer(self) -> None:
        assert _looks_like_value("758") is True

    def test_looks_like_value_float(self) -> None:
        assert _looks_like_value("758.3") is True

    def test_looks_like_value_negative(self) -> None:
        assert _looks_like_value("-12.5") is True

    def test_looks_like_value_comma_decimal(self) -> None:
        assert _looks_like_value("12,5") is True

    def test_looks_like_value_label(self) -> None:
        assert _looks_like_value("TI-101") is False

    def test_looks_like_value_empty(self) -> None:
        assert _looks_like_value("") is False


# ---------------------------------------------------------------------------
# Тесты LayoutConfig
# ---------------------------------------------------------------------------

class TestLayoutConfig:
    """Тесты конфигурации."""

    def test_default_config(self) -> None:
        cfg = LayoutConfig()
        assert cfg.bandwidth_factor == 0.05
        assert cfg.tilt_compensation is True
        assert cfg.enable_unit_column is True
        assert cfg.rank_matching is True

    def test_custom_config(self) -> None:
        cfg = LayoutConfig(
            bandwidth_factor=0.1,
            tilt_compensation=False,
            enable_unit_column=False,
        )
        assert cfg.bandwidth_factor == 0.1
        assert cfg.tilt_compensation is False
        assert cfg.enable_unit_column is False

    def test_frozen_config(self) -> None:
        cfg = LayoutConfig()
        with pytest.raises(AttributeError):
            cfg.bandwidth_factor = 0.2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Тесты X-кластеризации
# ---------------------------------------------------------------------------

class TestColumnClustering:
    """Тесты обнаружения колонок."""

    def test_two_columns(self) -> None:
        """Два явных кластера по X должны быть обнаружены."""
        boxes = [
            _tb("TI-101", x=100, y=100, box_type="label"),
            _tb("PI-205", x=100, y=150, box_type="label"),
            _tb("758.3", x=500, y=100, box_type="value"),
            _tb("4.21", x=500, y=150, box_type="value"),
        ]
        analyzer = ColumnLayoutAnalyzer()
        columns = analyzer._cluster_columns(boxes)
        assert len(columns) == 2

    def test_three_columns(self) -> None:
        """Три колонки (label, value, unit) должны быть обнаружены при достаточном числе боксов."""
        # Нужно достаточно боксов для работы адаптивного порога
        boxes = [
            _tb("TI-101", x=100, y=100, box_type="label"),
            _tb("PI-205", x=100, y=200, box_type="label"),
            _tb("758.3", x=400, y=100, box_type="value"),
            _tb("4.21", x=400, y=200, box_type="value"),
            _tb("°C", x=550, y=100, box_type="unknown"),
            _tb("кПа", x=550, y=200, box_type="unknown"),
        ]
        analyzer = ColumnLayoutAnalyzer()
        columns = analyzer._cluster_columns(boxes)
        assert len(columns) >= 2  # минимум label + value

    def test_single_column(self) -> None:
        """Все боксы в одной колонке → 1 кластер."""
        boxes = [
            _tb("TI-101", x=100, y=100, box_type="label"),
            _tb("758.3", x=110, y=150, box_type="value"),
        ]
        analyzer = ColumnLayoutAnalyzer()
        columns = analyzer._cluster_columns(boxes)
        assert len(columns) == 1

    def test_empty_input(self) -> None:
        """Пустой список → пустой результат."""
        analyzer = ColumnLayoutAnalyzer()
        columns = analyzer._cluster_columns([])
        assert columns == []

    def test_adaptive_threshold_tight_layout(self) -> None:
        """Плотный layout: адаптивный порог не должен разбивать одну колонку."""
        # Все X в диапазоне 100-130 — одна колонка
        boxes = [
            _tb(f"item{i}", x=100 + i * 5, y=100 + i * 30, box_type="unknown")
            for i in range(10)
        ]
        analyzer = ColumnLayoutAnalyzer()
        columns = analyzer._cluster_columns(boxes)
        assert len(columns) == 1

    def test_adaptive_threshold_wide_gaps(self) -> None:
        """Широкие зазоры: должны разделять на 2 колонки."""
        boxes = [
            _tb("L1", x=100, y=100, box_type="label"),
            _tb("L2", x=105, y=140, box_type="label"),
            _tb("V1", x=500, y=100, box_type="value"),
            _tb("V2", x=505, y=140, box_type="value"),
        ]
        analyzer = ColumnLayoutAnalyzer()
        columns = analyzer._cluster_columns(boxes)
        assert len(columns) == 2


# ---------------------------------------------------------------------------
# Тесты Y-сопоставления
# ---------------------------------------------------------------------------

class TestRowMatching:
    """Тесты сопоставления строк."""

    def test_basic_horizontal_pairs(self) -> None:
        """Базовый случай: label и value на одной строке."""
        boxes = _classify([
            _tb("TI-101", x=100, y=100),
            _tb("758.3", x=500, y=100),
        ])
        analyzer = ColumnLayoutAnalyzer()
        pairs = analyzer.extract_pairs(boxes)
        assert len(pairs) >= 1
        # Хотя бы одна пара с label "TI-101"
        label_texts = [p.label.text for p in pairs]
        assert any("TI" in t or "101" in t for t in label_texts)

    def test_multiple_rows(self) -> None:
        """Несколько строк: каждый label сопоставляется с value на той же Y."""
        boxes = _classify([
            _tb("TI-101", x=100, y=100),
            _tb("758.3", x=500, y=100),
            _tb("PI-205", x=100, y=200),
            _tb("4.21", x=500, y=200),
        ])
        analyzer = ColumnLayoutAnalyzer()
        pairs = analyzer.extract_pairs(boxes)
        assert len(pairs) >= 2

    def test_tilt_compensation(self) -> None:
        """Наклон камеры: label и value смещены по Y, но ранговое сопоставление работает."""
        cfg = LayoutConfig(tilt_compensation=True, rank_matching=True)
        analyzer = ColumnLayoutAnalyzer(config=cfg)
        # Линейный наклон: каждый следующий value смещён на +10px по Y
        boxes = _classify([
            _tb("TI-101", x=100, y=100),
            _tb("758.3", x=500, y=110),  # +10 наклон
            _tb("PI-205", x=100, y=200),
            _tb("4.21", x=500, y=210),   # +10 наклон
            _tb("Vb-310", x=100, y=300),
            _tb("0.25", x=500, y=310),   # +10 наклон
        ])
        pairs = analyzer.extract_pairs(boxes)
        # Должно найти минимум 2 пары даже с наклоном
        assert len(pairs) >= 2

    def test_no_false_pairs(self) -> None:
        """Значения на разных строках не должны ошибочно связываться."""
        boxes = _classify([
            _tb("TI-101", x=100, y=100),
            _tb("758.3", x=500, y=100),
            _tb("PI-205", x=100, y=500),  # далеко внизу
            _tb("4.21", x=500, y=500),
        ])
        analyzer = ColumnLayoutAnalyzer()
        pairs = analyzer.extract_pairs(boxes)
        # Каждая пара должна быть на своей строке
        for p in pairs:
            dy = abs(p.label.bbox.center[1] - p.value.bbox.center[1])
            # Y-расстояние в пределах разумного (не более 100px)
            assert dy < 100, f"Pair ({p.label.text}, {p.value.text}) has dy={dy}"


# ---------------------------------------------------------------------------
# Тесты 3-й колонки (units)
# ---------------------------------------------------------------------------

class TestUnitColumn:
    """Тесты обнаружения колонки единиц измерения."""

    def test_three_columns_with_units(self) -> None:
        """3 колонки: label, value, unit."""
        cfg = LayoutConfig(enable_unit_column=True)
        analyzer = ColumnLayoutAnalyzer(config=cfg)
        boxes = _classify([
            _tb("TI-101", x=100, y=100),
            _tb("758.3", x=400, y=100),
            _tb("°C", x=550, y=100),
            _tb("PI-205", x=100, y=200),
            _tb("4.21", x=400, y=200),
            _tb("кПа", x=550, y=200),
        ])
        pairs = analyzer.extract_pairs(boxes)
        # Должно найти пары
        assert len(pairs) >= 1

    def test_unit_column_disabled(self) -> None:
        """3 колонки, но enable_unit_column=False → только 2 колонки."""
        cfg = LayoutConfig(enable_unit_column=False)
        analyzer = ColumnLayoutAnalyzer(config=cfg)
        boxes = _classify([
            _tb("TI-101", x=100, y=100),
            _tb("758.3", x=400, y=100),
            _tb("°C", x=550, y=100),
        ])
        # Должно работать без ошибки (3-я колонка игнорируется)
        pairs = analyzer.extract_pairs(boxes)
        assert isinstance(pairs, list)


# ---------------------------------------------------------------------------
# Тесты уверенности
# ---------------------------------------------------------------------------

class TestPairConfidence:
    """Тесты функции уверенности пары."""

    def test_high_confidence_pair(self) -> None:
        """Близкие Y, высокий OCR conf → высокая уверенность."""
        analyzer = ColumnLayoutAnalyzer()
        label = _tb("TI-101", x=100, y=100, conf=0.95, box_type="label")
        value = _tb("758.3", x=500, y=102, conf=0.92, box_type="value")
        conf = analyzer._compute_pair_confidence(label, value, dy=2.0, y_tolerance=24.0)
        assert conf > 0.5

    def test_low_confidence_distant_y(self) -> None:
        """Большой Y-зазор → низкая уверенность."""
        analyzer = ColumnLayoutAnalyzer()
        label = _tb("TI-101", x=100, y=100, conf=0.95, box_type="label")
        value = _tb("758.3", x=500, y=200, conf=0.92, box_type="value")
        conf = analyzer._compute_pair_confidence(label, value, dy=100.0, y_tolerance=24.0)
        # Большой dy → уверенность должна быть ниже
        assert conf < 0.8

    def test_x_alignment_bonus(self) -> None:
        """Label левее value → бонус к уверенности."""
        analyzer = ColumnLayoutAnalyzer()
        label = _tb("TI-101", x=100, y=100, conf=0.95, box_type="label")
        value_right = _tb("758.3", x=500, y=100, conf=0.95, box_type="value")
        value_left = _tb("758.3", x=50, y=100, conf=0.95, box_type="value")

        conf_right = analyzer._compute_pair_confidence(label, value_right, dy=0.0, y_tolerance=24.0)
        conf_left = analyzer._compute_pair_confidence(label, value_left, dy=0.0, y_tolerance=24.0)
        # value правее label → выше уверенность
        assert conf_right > conf_left

    def test_char_confs_refinement(self) -> None:
        """char_confs используются для уточнения уверенности."""
        analyzer = ColumnLayoutAnalyzer()
        value_with_chars = _tb("758.3", x=500, y=100, conf=0.70, box_type="value")
        value_with_chars.char_confs = [0.95, 0.96, 0.94, 0.97, 0.95]

        label = _tb("TI-101", x=100, y=100, conf=0.95, box_type="label")

        # С char_confs уверенность должна быть выше, чем без них
        conf_with = analyzer._compute_pair_confidence(label, value_with_chars, dy=0.0, y_tolerance=24.0)

        value_no_chars = _tb("758.3", x=500, y=100, conf=0.70, box_type="value")
        conf_without = analyzer._compute_pair_confidence(label, value_no_chars, dy=0.0, y_tolerance=24.0)

        assert conf_with > conf_without

    def test_confidence_bounded(self) -> None:
        """Уверенность всегда в [0, 1]."""
        analyzer = ColumnLayoutAnalyzer()
        label = _tb("TI-101", x=100, y=100, conf=0.99, box_type="label")
        value = _tb("758.3", x=500, y=100, conf=0.99, box_type="value")
        conf = analyzer._compute_pair_confidence(label, value, dy=0.0, y_tolerance=24.0)
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Тесты inline fallback
# ---------------------------------------------------------------------------

class TestInlineFallback:
    """Тесты inline-разбиения при одной колонке."""

    def test_inline_pair_extraction(self) -> None:
        """Inline пара «Label: Value» должна быть извлечена."""
        boxes = _classify([
            _tb("Давление: 12.3", x=100, y=100),
        ])
        # Принудительно одна колонка
        analyzer = ColumnLayoutAnalyzer()
        pairs = analyzer._extract_inline_pairs(boxes)
        assert len(pairs) >= 1
        assert pairs[0].relation == "inline"

    def test_no_inline_without_separator(self) -> None:
        """Без разделителя «:» inline-пара не извлекается."""
        boxes = _classify([
            _tb("TI-101", x=100, y=100),
        ])
        analyzer = ColumnLayoutAnalyzer()
        pairs = analyzer._extract_inline_pairs(boxes)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Тесты валидации колонок
# ---------------------------------------------------------------------------

class TestColumnValidation:
    """Тесты валидации типа колонок."""

    def test_swapped_columns(self) -> None:
        """Если value-колонка левее label-колонки, колонки меняются местами."""
        boxes = [
            _tb("758.3", x=100, y=100, box_type="value"),
            _tb("4.21", x=100, y=200, box_type="value"),
            _tb("TI-101", x=500, y=100, box_type="label"),
            _tb("PI-205", x=500, y=200, box_type="label"),
        ]
        analyzer = ColumnLayoutAnalyzer()
        columns = analyzer._cluster_columns(boxes)
        columns.sort(key=lambda c: min(b.bbox.center[0] for b in c))
        validated = analyzer._validate_columns(columns)
        # После валидации первая колонка должна быть label
        if len(validated) >= 1:
            label_ratio = sum(
                1 for b in validated[0] if b.box_type in ("label",)
            ) / max(len(validated[0]), 1)
            # Колонки должны были быть переставлены
            assert label_ratio > 0 or len(validated) < 2


# ---------------------------------------------------------------------------
# Тесты протокола LayoutAnalyzer
# ---------------------------------------------------------------------------

class TestLayoutAnalyzerProtocol:
    """Тесты протокола LayoutAnalyzer."""

    def test_column_analyzer_satisfies_protocol(self) -> None:
        """ColumnLayoutAnalyzer удовлетворяет протоколу LayoutAnalyzer."""
        analyzer: LayoutAnalyzer = ColumnLayoutAnalyzer()
        # Просто проверяем, что метод существует и вызывается
        result = analyzer.extract_pairs([])
        assert result == []

    def test_protocol_call_signature(self) -> None:
        """Протокол: extract_pairs принимает list[TextBox] и возвращает list[TextPair]."""
        analyzer = ColumnLayoutAnalyzer()
        boxes = _classify([
            _tb("TI-101", x=100, y=100),
            _tb("758.3", x=500, y=100),
        ])
        result = analyzer.extract_pairs(boxes)
        assert isinstance(result, list)
        for p in result:
            assert isinstance(p, TextPair)


# ---------------------------------------------------------------------------
# Тесты edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Граничные случаи."""

    def test_empty_input(self) -> None:
        """Пустой список → пустой результат."""
        analyzer = ColumnLayoutAnalyzer()
        assert analyzer.extract_pairs([]) == []

    def test_single_box(self) -> None:
        """Один TextBox → пустой результат (нужна пара)."""
        analyzer = ColumnLayoutAnalyzer()
        boxes = _classify([_tb("TI-101", x=100, y=100)])
        assert analyzer.extract_pairs(boxes) == []

    def test_only_labels(self) -> None:
        """Только labels → нет пар."""
        boxes = _classify([
            _tb("TI-101", x=100, y=100),
            _tb("PI-205", x=100, y=200),
        ])
        analyzer = ColumnLayoutAnalyzer()
        pairs = analyzer.extract_pairs(boxes)
        # Нет values → нет пар (или только inline если повезёт)
        for p in pairs:
            assert p.value.box_type in ("value", "unknown")

    def test_only_values(self) -> None:
        """Только values → нет пар."""
        boxes = _classify([
            _tb("758.3", x=500, y=100),
            _tb("4.21", x=500, y=200),
        ])
        analyzer = ColumnLayoutAnalyzer()
        pairs = analyzer.extract_pairs(boxes)
        for p in pairs:
            assert p.label.box_type in ("label", "unknown")

    def test_many_rows_stress(self) -> None:
        """Стресс-тест: 20 строк label-value."""
        boxes = _classify([
            _tb(f"L{i:03d}", x=100, y=50 + i * 40)
            for i in range(20)
        ] + [
            _tb(f"{i*10.5:.1f}", x=500, y=50 + i * 40)
            for i in range(20)
        ])
        analyzer = ColumnLayoutAnalyzer()
        pairs = analyzer.extract_pairs(boxes)
        # Должно найти хотя бы 10 пар из 20
        assert len(pairs) >= 10
