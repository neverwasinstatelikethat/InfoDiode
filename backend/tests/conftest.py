"""Общие фикстуры для тестов."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Создаёт тестовый кадр 640x480 BGR с текстовыми блоками."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Светло-серый фон

    # Добавляем «заголовок»
    frame[0:40, :] = 200

    # «Левая навигация»
    frame[40:440, 0:80] = 180

    # «Правая панель»
    frame[40:440, 500:] = 190

    # «Нижняя панель»
    frame[440:, :] = 200

    return frame


@pytest.fixture
def sample_ocr_results():
    """Возвращает предзаготовленные результаты OCR для тестирования кластеризации."""
    from app.models.schemas import BoundingBox, OCRTextResult

    return [
        OCRTextResult(text="TI-101", confidence=0.95, bbox=BoundingBox(x1=0.2, y1=0.3, x2=0.25, y2=0.33)),
        OCRTextResult(text="758.3", confidence=0.92, bbox=BoundingBox(x1=0.2, y1=0.34, x2=0.25, y2=0.37)),
        OCRTextResult(text="PI-205", confidence=0.90, bbox=BoundingBox(x1=0.4, y1=0.5, x2=0.45, y2=0.53)),
        OCRTextResult(text="4.21", confidence=0.88, bbox=BoundingBox(x1=0.4, y1=0.54, x2=0.45, y2=0.57)),
        OCRTextResult(text="Vb-310", confidence=0.85, bbox=BoundingBox(x1=0.6, y1=0.7, x2=0.65, y2=0.73)),
        OCRTextResult(text="0.25", confidence=0.91, bbox=BoundingBox(x1=0.6, y1=0.74, x2=0.65, y2=0.77)),
    ]


@pytest.fixture
def sample_parameter_table() -> list[dict]:
    """Возвращает тестовую таблицу параметров."""
    return [
        {"id": 1, "name": "Температура газа (TI-101)", "unit": "°С", "short_name": "T", "decimal_places": 1},
        {"id": 2, "name": "Давление на выходе (PI-205)", "unit": "кПа", "short_name": "P", "decimal_places": 2},
        {"id": 3, "name": "Вибрация подшипника (Vb-310)", "unit": "мм/с", "short_name": "Vb", "decimal_places": 2},
    ]


@pytest.fixture
def sample_xml() -> str:
    """Возвращает пример XML в правильном формате."""
    return '''<sheme id="54d11679-95db-4b8b-9b8d-ab11f251ef38">
<parameters timestamp = "22:53:00.001">
    <param id="1">758.3</param>
    <param id="2">4.21</param>
    <param id="3">0.25</param>
</parameters>
<parameters timestamp = "22:53:00.501">
    <param id="1">758.3</param>
    <param id="2">4.22</param>
    <param id="3">0.26</param>
</parameters>
</sheme>'''
