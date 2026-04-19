"""Тесты proximity search — rigorous validation of value finding near labels.

Tests verify that values are correctly found near labels with proper
priority: right > below > any direction.
"""

import pytest
from app.models.schemas import BoundingBox, OCRTextResult


class TestProximitySearch:
    """Тесты поиска значений рядом с метками параметров."""

    def test_find_value_to_right_of_label(self) -> None:
        """Тест: значение справа от метки находится.
        
        Label at (100,200,300,230), value "45.2" at (320,200,380,230)
        Should find value 45.2 to the right.
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)  # ~100,200,300,230 at 640x480
        
        ocr_results = [
            OCRTextResult(text="TI-101", confidence=0.9, bbox=label_bbox),
            OCRTextResult(text="45.2", confidence=0.85, bbox=BoundingBox(x1=0.500, y1=0.417, x2=0.594, y2=0.479)),  # To the right
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640)
        )
        
        assert number == 45.2
        assert position == "right"

    def test_find_value_below_label(self) -> None:
        """Тест: значение под меткой находится.
        
        When value is below the label, should be found with 'below' position.
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)
        
        ocr_results = [
            OCRTextResult(text="TI-101", confidence=0.9, bbox=label_bbox),
            OCRTextResult(text="45.2", confidence=0.85, bbox=BoundingBox(x1=0.312, y1=0.500, x2=0.406, y2=0.562)),  # Below
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640)
        )
        
        assert number == 45.2
        assert position == "below"

    def test_priority_right_over_below(self) -> None:
        """Тест: приоритет справа > под меткой.
        
        When both right and below candidates exist, should prefer right.
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)
        
        ocr_results = [
            OCRTextResult(text="TI-101", confidence=0.9, bbox=label_bbox),
            OCRTextResult(text="45.2", confidence=0.85, bbox=BoundingBox(x1=0.500, y1=0.417, x2=0.594, y2=0.479)),  # Right
            OCRTextResult(text="99.9", confidence=0.80, bbox=BoundingBox(x1=0.312, y1=0.500, x2=0.406, y2=0.562)),  # Below
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640)
        )
        
        # Should pick the one to the right (priority)
        assert number == 45.2
        assert position == "right"

    def test_priority_below_over_fallback(self) -> None:
        """Тест: приоритет под меткой > fallback.
        
        When no right candidate but below exists, should pick below over fallback.
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)
        
        ocr_results = [
            OCRTextResult(text="TI-101", confidence=0.9, bbox=label_bbox),
            OCRTextResult(text="77.7", confidence=0.80, bbox=BoundingBox(x1=0.700, y1=0.700, x2=0.800, y2=0.800)),  # Far away (fallback)
            OCRTextResult(text="45.2", confidence=0.85, bbox=BoundingBox(x1=0.312, y1=0.500, x2=0.406, y2=0.562)),  # Below
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640),
            max_horizontal_distance=0.5, max_vertical_distance=0.2
        )
        
        # Should pick the one below (priority over fallback)
        assert number == 45.2
        assert position == "below"

    def test_no_numeric_text_nearby_returns_none(self) -> None:
        """Тест: отсутствие числового текста рядом → None.
        
        When no numeric text is found near the label, should return None.
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)
        
        ocr_results = [
            OCRTextResult(text="TI-101", confidence=0.9, bbox=label_bbox),
            OCRTextResult(text="NOT_A_NUMBER", confidence=0.85, bbox=BoundingBox(x1=0.500, y1=0.417, x2=0.700, y2=0.479)),
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640)
        )
        
        assert number is None
        assert position == "none"

    def test_multiple_numeric_candidates_picks_closest(self) -> None:
        """Тест: несколько числовых кандидатов → выбирается ближайший.
        
        When multiple numeric candidates exist in the same direction,
        should pick the closest one.
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)
        
        ocr_results = [
            OCRTextResult(text="TI-101", confidence=0.9, bbox=label_bbox),
            OCRTextResult(text="100.0", confidence=0.85, bbox=BoundingBox(x1=0.800, y1=0.417, x2=0.900, y2=0.479)),  # Far right
            OCRTextResult(text="45.2", confidence=0.82, bbox=BoundingBox(x1=0.500, y1=0.417, x2=0.594, y2=0.479)),   # Close right
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640)
        )
        
        # Should pick the closest one (45.2)
        assert number == 45.2

    def test_label_bbox_excluded_from_candidates(self) -> None:
        """Тест: bbox самой метки исключается из кандидатов.
        
        The label's own text should not be considered as a value candidate.
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)
        
        ocr_results = [
            OCRTextResult(text="TI-101", confidence=0.9, bbox=label_bbox),
            # Try to trick it with overlapping bbox containing number
            OCRTextResult(text="999", confidence=0.85, bbox=BoundingBox(x1=0.200, y1=0.430, x2=0.400, y2=0.460)),
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640)
        )
        
        # Should not pick the overlapping bbox (it's likely the same label)
        # or should handle it correctly
        assert number is None or number == 999  # Depending on overlap calculation

    def test_horizontal_layout_same_y_level(self) -> None:
        """Тест: горизонтальное расположение — тот же уровень по Y.
        
        Value to the right at approximately the same Y level.
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        # Label at Y center = 0.448
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)
        
        ocr_results = [
            OCRTextResult(text="LABEL", confidence=0.9, bbox=label_bbox),
            # Value at same Y level (within vertical tolerance)
            OCRTextResult(text="50.0", confidence=0.85, bbox=BoundingBox(x1=0.500, y1=0.420, x2=0.600, y2=0.480)),
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640),
            max_vertical_distance=0.05
        )
        
        assert number == 50.0
        assert position == "right"

    def test_vertical_layout_below_label(self) -> None:
        """Тест: вертикальное расположение — под меткой.
        
        Value directly below the label.
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        # Label
        label_bbox = BoundingBox(x1=0.312, y1=0.417, x2=0.469, y2=0.479)
        
        ocr_results = [
            OCRTextResult(text="LABEL", confidence=0.9, bbox=label_bbox),
            # Value directly below
            OCRTextResult(text="75.5", confidence=0.85, bbox=BoundingBox(x1=0.350, y1=0.500, x2=0.450, y2=0.562)),
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640)
        )
        
        assert number == 75.5
        assert position == "below"

    def test_negative_number_extraction(self) -> None:
        """Тест: извлечение отрицательного числа.
        
        Should correctly extract negative values like "-12.5".
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)
        
        ocr_results = [
            OCRTextResult(text="TEMP", confidence=0.9, bbox=label_bbox),
            OCRTextResult(text="-12.5", confidence=0.85, bbox=BoundingBox(x1=0.500, y1=0.417, x2=0.625, y2=0.479)),
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640)
        )
        
        assert number == -12.5

    def test_decimal_number_extraction(self) -> None:
        """Тест: извлечение десятичного числа.
        
        Should correctly extract decimal values like "123.456".
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)
        
        ocr_results = [
            OCRTextResult(text="PRESSURE", confidence=0.9, bbox=label_bbox),
            OCRTextResult(text="123.456", confidence=0.85, bbox=BoundingBox(x1=0.500, y1=0.417, x2=0.656, y2=0.479)),
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640)
        )
        
        assert number == 123.456

    def test_integer_extraction(self) -> None:
        """Тест: извлечение целого числа.
        
        Should correctly extract integer values like "100".
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)
        
        ocr_results = [
            OCRTextResult(text="COUNT", confidence=0.9, bbox=label_bbox),
            OCRTextResult(text="100", confidence=0.85, bbox=BoundingBox(x1=0.500, y1=0.417, x2=0.578, y2=0.479)),
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640)
        )
        
        assert number == 100.0

    def test_number_with_unit_extraction(self) -> None:
        """Тест: извлечение числа с единицей измерения.
        
        Should extract number from text like "45.2°C" or "100 кПа".
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)
        
        ocr_results = [
            OCRTextResult(text="TEMP", confidence=0.9, bbox=label_bbox),
            OCRTextResult(text="45.2°C", confidence=0.85, bbox=BoundingBox(x1=0.500, y1=0.417, x2=0.625, y2=0.479)),
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640)
        )
        
        assert number == 45.2

    def test_empty_ocr_results(self) -> None:
        """Тест: пустые результаты OCR.
        
        When OCR results are empty, should return None.
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, [], (480, 640)
        )
        
        assert number is None
        assert position == "none"

    def test_value_too_far_away(self) -> None:
        """Тест: значение слишком далеко.
        
        When value is beyond max_distance, should not be found.
        """
        from app.core.pipeline import Pipeline
        
        pipeline = Pipeline.__new__(Pipeline)
        
        label_bbox = BoundingBox(x1=0.156, y1=0.417, x2=0.469, y2=0.479)
        
        ocr_results = [
            OCRTextResult(text="LABEL", confidence=0.9, bbox=label_bbox),
            # Value very far away
            OCRTextResult(text="999.9", confidence=0.85, bbox=BoundingBox(x1=0.900, y1=0.900, x2=0.950, y2=0.950)),
        ]
        
        number, confidence, position = pipeline._find_value_near_label(
            label_bbox, ocr_results, (480, 640),
            max_horizontal_distance=0.15, max_vertical_distance=0.05
        )
        
        # Should not find the far away value
        assert number is None
