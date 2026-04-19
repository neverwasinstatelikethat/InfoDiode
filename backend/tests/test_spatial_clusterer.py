"""Тесты пространственного кластеризатора."""

from app.core.spatial_clusterer import cluster_label_value_pairs, parse_right_panel
from app.models.schemas import BoundingBox, OCRTextResult


class TestSpatialClusterer:
    """Тесты связывания label-value пар."""

    def test_cluster_basic_pairs(self, sample_ocr_results) -> None:
        """Тест базовой кластеризации label-value пар."""
        pairs = cluster_label_value_pairs(sample_ocr_results, (480, 640))

        assert len(pairs) >= 2

        # Проверяем, что метки связаны с правильными значениями
        for pair in pairs:
            if "TI" in pair.label:
                assert pair.value == "758.3"
            elif "PI" in pair.label:
                assert pair.value == "4.21"
            elif "Vb" in pair.label:
                assert pair.value == "0.25"

    def test_cluster_empty_input(self) -> None:
        """Тест кластеризации пустого входа."""
        pairs = cluster_label_value_pairs([], (480, 640))
        assert len(pairs) == 0

    def test_cluster_values_only(self) -> None:
        """Тест кластеризации только значений (без меток). не должно быть пар."""
        results = [
            OCRTextResult(text="758.3", confidence=0.9, bbox=BoundingBox(x1=0.2, y1=0.3, x2=0.25, y2=0.33)),
            OCRTextResult(text="4.21", confidence=0.8, bbox=BoundingBox(x1=0.4, y1=0.5, x2=0.45, y2=0.53)),
        ]
        pairs = cluster_label_value_pairs(results, (480, 640))
        assert len(pairs) == 0  # Нет меток = нет пар

    def test_right_panel_parsing(self) -> None:
        """Тест парсинга правой панели."""
        ocr_results = [
            # Левая часть (метки)
            OCRTextResult(text="Давление", confidence=0.9, bbox=BoundingBox(x1=0.82, y1=0.3, x2=0.88, y2=0.32)),
            OCRTextResult(text="Температура", confidence=0.85, bbox=BoundingBox(x1=0.82, y1=0.4, x2=0.88, y2=0.42)),
            # Правая часть (значения)
            OCRTextResult(text="4.2", confidence=0.92, bbox=BoundingBox(x1=0.9, y1=0.3, x2=0.95, y2=0.32)),
            OCRTextResult(text="65.1", confidence=0.88, bbox=BoundingBox(x1=0.9, y1=0.4, x2=0.95, y2=0.42)),
        ]

        zone_bbox = BoundingBox(x1=0.8, y1=0.2, x2=1.0, y2=0.8)
        pairs = parse_right_panel(ocr_results, zone_bbox, (480, 640))

        assert len(pairs) >= 1
        for pair in pairs:
            assert pair.zone.value == "right_panel"
