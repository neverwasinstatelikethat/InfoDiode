"""Тесты сегментатора зон."""

import numpy as np
import pytest

from app.core.zone_segmentor import (
    _build_zones_from_boundaries,
    _cluster_indices,
    _get_geometric_zones,
    _is_valid_segmentation,
    _ZoneBoundaries,
    detect_popup_dialog,
    segment_zones,
)
from app.models.schemas import BoundingBox, ZoneBoundary, ZoneType


class TestZoneSegmentor:
    """Тесты сегментации мнемосхемы на зоны."""

    def test_segment_zones_returns_all_five(self, sample_frame) -> None:
        """Тест: сегментация возвращает все 5 зон."""
        zones = segment_zones(sample_frame)

        assert len(zones) == 5

        zone_types = {z.zone for z in zones}
        assert ZoneType.HEADER in zone_types
        assert ZoneType.LEFT_NAV in zone_types
        assert ZoneType.CENTRAL_SCHEMA in zone_types
        assert ZoneType.RIGHT_PANEL in zone_types
        assert ZoneType.BOTTOM_BAR in zone_types

    def test_central_schema_is_largest(self, sample_frame) -> None:
        """Тест: центральная схема — самая большая зона."""
        zones = segment_zones(sample_frame)

        central = next(z for z in zones if z.zone == ZoneType.CENTRAL_SCHEMA)
        central_area = (central.bbox.x2 - central.bbox.x1) * (central.bbox.y2 - central.bbox.y1)

        for zone in zones:
            if zone.zone != ZoneType.CENTRAL_SCHEMA:
                area = (zone.bbox.x2 - zone.bbox.x1) * (zone.bbox.y2 - zone.bbox.y1)
                assert central_area >= area

    def test_zones_dont_overlap(self, sample_frame) -> None:
        """Тест: зоны не перекрываются (кроме границ)."""
        zones = segment_zones(sample_frame)

        for i, z1 in enumerate(zones):
            for j, z2 in enumerate(zones):
                if i >= j:
                    continue
                # Проверяем, что центр одной зоны не внутри другой
                cx = (z1.bbox.x1 + z1.bbox.x2) / 2
                cy = (z1.bbox.y1 + z1.bbox.y2) / 2
                # Это может быть не совсем точно для смежных зон, но центр не должен быть глубоко внутри
                # Проверяем что зоны имеют непересекающиеся внутренности
                # (допускаем совпадение границ)
                x_overlap = max(0, min(z1.bbox.x2, z2.bbox.x2) - max(z1.bbox.x1, z2.bbox.x1))
                y_overlap = max(0, min(z1.bbox.y2, z2.bbox.y2) - max(z1.bbox.y1, z2.bbox.y1))
                overlap_area = x_overlap * y_overlap
                area1 = (z1.bbox.x2 - z1.bbox.x1) * (z1.bbox.y2 - z1.bbox.y1)
                # Допускаем перекрытие только на границах (< 10% площади меньшей зоны)
                assert overlap_area < area1 * 0.1


class TestGeometricFallback:
    """Тесты геометрического fallback."""

    def test_geometric_zones_returns_five_zones(self) -> None:
        """Тест: геометрический fallback возвращает 5 зон."""
        zones, boundaries = _get_geometric_zones(480, 640)

        assert len(zones) == 5
        zone_types = {z.zone for z in zones}
        assert ZoneType.HEADER in zone_types
        assert ZoneType.LEFT_NAV in zone_types
        assert ZoneType.CENTRAL_SCHEMA in zone_types
        assert ZoneType.RIGHT_PANEL in zone_types
        assert ZoneType.BOTTOM_BAR in zone_types

    def test_geometric_zones_percentages(self) -> None:
        """Тест: геометрический fallback использует правильные проценты."""
        h, w = 1000, 1000
        zones, boundaries = _get_geometric_zones(h, w)

        # Header: 8% высоты
        assert boundaries.header_y == 80

        # Left nav: 12% ширины
        assert boundaries.left_nav_x == 120

        # Right panel: 22% ширины (от правого края)
        assert boundaries.right_panel_x == 780

        # Bottom: 6% высоты (от низа)
        assert boundaries.bottom_y == 940

    def test_geometric_central_zone_is_largest(self) -> None:
        """Тест: центральная зона самая большая при геометрическом fallback."""
        zones, _ = _get_geometric_zones(480, 640)

        central = next(z for z in zones if z.zone == ZoneType.CENTRAL_SCHEMA)
        central_area = (central.bbox.x2 - central.bbox.x1) * (central.bbox.y2 - central.bbox.y1)

        for zone in zones:
            if zone.zone != ZoneType.CENTRAL_SCHEMA:
                area = (zone.bbox.x2 - zone.bbox.x1) * (zone.bbox.y2 - zone.bbox.y1)
                assert central_area > area


class TestProjectionProfile:
    """Тесты метода профилей проекций."""

    @pytest.fixture
    def blank_frame(self) -> np.ndarray:
        """Создаёт пустой кадр без контента."""
        return np.ones((480, 640, 3), dtype=np.uint8) * 240

    @pytest.fixture
    def frame_with_clear_zones(self) -> np.ndarray:
        """Создаёт кадр с чёткими зонами (тёмные разделители)."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 240

        # Вертикальные разделители
        frame[:, 75:80] = 50   # Левый разделитель
        frame[:, 500:505] = 50  # Правый разделитель

        # Горизонтальные разделители
        frame[38:42, :] = 50   # Заголовок
        frame[450:455, :] = 50  # Низ

        return frame

    def test_segment_zones_projection_on_blank_frame(self, blank_frame) -> None:
        """Тест: профили проекций на пустом кадре."""
        from app.core.zone_segmentor import _segment_zones_projection

        gray = np.mean(blank_frame, axis=2).astype(np.uint8)
        zones, boundaries = _segment_zones_projection(gray, 480, 640)

        # Должен вернуть 5 зон
        assert len(zones) == 5

    def test_cluster_indices_empty(self) -> None:
        """Тест: кластеризация пустого массива."""
        result = _cluster_indices(np.array([], dtype=np.int64))
        assert result == []

    def test_cluster_indices_single(self) -> None:
        """Тест: кластеризация одного элемента."""
        result = _cluster_indices(np.array([5], dtype=np.int64))
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0] == 5

    def test_cluster_indices_cluster(self) -> None:
        """Тест: кластеризация смежных элементов."""
        # Смежные элементы должны быть в одном кластере
        result = _cluster_indices(np.array([10, 11, 12, 13, 14], dtype=np.int64))
        assert len(result) == 1
        assert len(result[0]) == 5

    def test_cluster_indices_multiple_clusters(self) -> None:
        """Тест: несколько отдельных кластеров."""
        # Два кластера с разрывом > 2
        result = _cluster_indices(np.array([10, 11, 12, 20, 21, 22], dtype=np.int64))
        assert len(result) == 2
        assert len(result[0]) == 3  # Первый кластер
        assert len(result[1]) == 3  # Второй кластер


class TestPopupDialogDetection:
    """Тесты детекции всплывающих диалогов."""

    @pytest.fixture
    def frame_with_popup(self) -> np.ndarray:
        """Создаёт кадр со всплывающим диалогом."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100  # Тёмный фон

        # Добавляем светлый диалог
        frame[100:380, 150:490] = 240  # Белый прямоугольник

        # Добавляем рамку диалога (тёмные края)
        frame[100:105, 150:490] = 50   # Верхняя рамка
        frame[375:380, 150:490] = 50   # Нижняя рамка
        frame[100:380, 150:155] = 50   # Левая рамка
        frame[100:380, 485:490] = 50   # Правая рамка

        return frame

    @pytest.fixture
    def frame_without_popup(self) -> np.ndarray:
        """Создаёт кадр без диалога."""
        return np.ones((480, 640, 3), dtype=np.uint8) * 150

    def test_detect_popup_dialog_finds_popup(self, frame_with_popup) -> None:
        """Тест: детекция находит всплывающий диалог."""
        result = detect_popup_dialog(frame_with_popup)

        assert result.detected is True
        assert result.bbox is not None
        assert result.mask is not None

        # Проверяем, что bbox примерно соответствует диалогу
        # (с допуском на морфологические операции)
        assert result.bbox.x1 < 0.3  # Начало примерно 150/640
        assert result.bbox.x2 > 0.7  # Конец примерно 490/640
        assert result.bbox.y1 < 0.3  # Начало примерно 100/480
        assert result.bbox.y2 > 0.7  # Конец примерно 380/480

    def test_detect_popup_dialog_no_popup(self, frame_without_popup) -> None:
        """Тест: детекция не находит диалог там, где его нет."""
        result = detect_popup_dialog(frame_without_popup)

        assert result.detected is False
        assert result.bbox is None
        assert result.mask is None

    def test_popup_too_small_not_detected(self) -> None:
        """Тест: слишком маленький диалог не детектируется."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

        # Очень маленький диалог (< 15% площади)
        frame[100:150, 100:150] = 240

        result = detect_popup_dialog(frame)
        assert result.detected is False


class TestValidation:
    """Тесты валидации сегментации."""

    def test_is_valid_segmentation_five_zones(self) -> None:
        """Тест: валидная сегментация с 5 зонами."""
        zones = [
            ZoneBoundary(zone=ZoneType.HEADER, bbox=BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=0.08)),
            ZoneBoundary(zone=ZoneType.LEFT_NAV, bbox=BoundingBox(x1=0.0, y1=0.08, x2=0.12, y2=0.94)),
            ZoneBoundary(zone=ZoneType.CENTRAL_SCHEMA, bbox=BoundingBox(x1=0.12, y1=0.08, x2=0.78, y2=0.94)),
            ZoneBoundary(zone=ZoneType.RIGHT_PANEL, bbox=BoundingBox(x1=0.78, y1=0.08, x2=1.0, y2=0.94)),
            ZoneBoundary(zone=ZoneType.BOTTOM_BAR, bbox=BoundingBox(x1=0.0, y1=0.94, x2=1.0, y2=1.0)),
        ]

        assert _is_valid_segmentation(zones, 480, 640) is True

    def test_is_valid_segmentation_wrong_count(self) -> None:
        """Тест: невалидная сегментация с неверным количеством зон."""
        zones = [
            ZoneBoundary(zone=ZoneType.CENTRAL_SCHEMA, bbox=BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)),
        ]

        assert _is_valid_segmentation(zones, 480, 640) is False

    def test_is_valid_segmentation_central_too_small(self) -> None:
        """Тест: невалидная сегментация с маленькой центральной зоной."""
        zones = [
            ZoneBoundary(zone=ZoneType.HEADER, bbox=BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=0.45)),
            ZoneBoundary(zone=ZoneType.LEFT_NAV, bbox=BoundingBox(x1=0.0, y1=0.45, x2=0.45, y2=0.55)),
            ZoneBoundary(zone=ZoneType.CENTRAL_SCHEMA, bbox=BoundingBox(x1=0.45, y1=0.45, x2=0.55, y2=0.55)),  # < 30%
            ZoneBoundary(zone=ZoneType.RIGHT_PANEL, bbox=BoundingBox(x1=0.55, y1=0.45, x2=1.0, y2=0.55)),
            ZoneBoundary(zone=ZoneType.BOTTOM_BAR, bbox=BoundingBox(x1=0.0, y1=0.55, x2=1.0, y2=1.0)),
        ]

        assert _is_valid_segmentation(zones, 480, 640) is False

    def test_is_valid_segmentation_zero_area(self) -> None:
        """Тест: невалидная сегментация с нулевой площадью зоны."""
        zones = [
            ZoneBoundary(zone=ZoneType.HEADER, bbox=BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=0.08)),
            ZoneBoundary(zone=ZoneType.LEFT_NAV, bbox=BoundingBox(x1=0.0, y1=0.08, x2=0.12, y2=0.94)),
            ZoneBoundary(zone=ZoneType.CENTRAL_SCHEMA, bbox=BoundingBox(x1=0.12, y1=0.08, x2=0.78, y2=0.94)),
            ZoneBoundary(zone=ZoneType.RIGHT_PANEL, bbox=BoundingBox(x1=0.5, y1=0.08, x2=0.5, y2=0.94)),  # Нулевая ширина
            ZoneBoundary(zone=ZoneType.BOTTOM_BAR, bbox=BoundingBox(x1=0.0, y1=0.94, x2=1.0, y2=1.0)),
        ]

        assert _is_valid_segmentation(zones, 480, 640) is False


class TestBuildZonesFromBoundaries:
    """Тесты построения зон из границ."""

    def test_build_zones_normalizes_coordinates(self) -> None:
        """Тест: координаты нормализуются в диапазон [0, 1]."""
        boundaries = _ZoneBoundaries(
            left_nav_x=120,
            right_panel_x=780,
            header_y=80,
            bottom_y=940,
        )

        zones = _build_zones_from_boundaries(boundaries, 1000, 1000)

        for zone in zones:
            assert 0.0 <= zone.bbox.x1 <= 1.0
            assert 0.0 <= zone.bbox.y1 <= 1.0
            assert 0.0 <= zone.bbox.x2 <= 1.0
            assert 0.0 <= zone.bbox.y2 <= 1.0

    def test_build_zones_returns_five_zones(self) -> None:
        """Тест: возвращает ровно 5 зон."""
        boundaries = _ZoneBoundaries(
            left_nav_x=120,
            right_panel_x=780,
            header_y=80,
            bottom_y=940,
        )

        zones = _build_zones_from_boundaries(boundaries, 1000, 1000)
        assert len(zones) == 5
