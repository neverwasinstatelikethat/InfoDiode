"""Сегментатор зон мнемосхемы SCADA.

Обнаруживает 5 зон на экране SCADA:
- Header (верхняя панель)
- Left Nav (левая навигация)
- Central Schema (основная область — PRIMARY)
- Right Panel (детальная информация — SECONDARY)
- Bottom Bar (нижняя строка состояния)

Использует каскад методов сегментации:
1. Линейная детекция (Canny + морфология)
2. Профили проекций (анализ плотности)
3. Геометрический fallback (захардкоженные проценты)
4. Детекция всплывающих диалогов
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.models.schemas import BoundingBox, ZoneBoundary, ZoneType

logger = logging.getLogger(__name__)

# Глобальный кэш зон: ключ = (height, width) — размеры кадра
# Инвалидируется при смене сцены через invalidate_zone_cache()
_zone_cache: dict[tuple[int, int], list[ZoneBoundary]] = {}


@dataclass
class _ZoneBoundaries:
    """Внутренняя структура для хранения границ зон в пикселях.

    Атрибуты:
        left_nav_x: Правая граница левой навигационной панели.
        right_panel_x: Левая граница правой панели.
        header_y: Нижняя граница заголовка.
        bottom_y: Верхняя граница нижней панели.
    """
    left_nav_x: int
    right_panel_x: int
    header_y: int
    bottom_y: int


def invalidate_zone_cache() -> None:
    """Сбрасывает кэш зон (вызывать при смене сцены/вкладки).

    При смене мнемосхемы SCADA зоны могут измениться,
    поэтому кэш нужно инвалилидировать.
    """
    global _zone_cache
    _zone_cache.clear()
    logger.debug("Кэш зон сброшен")


def segment_zones(frame: np.ndarray) -> list[ZoneBoundary]:
    """Разделяет кадр на зоны мнемосхемы SCADA.

    Использует каскад методов сегментации:
    1. Линейная детекция (Canny + морфология)
    2. Профили проекций (если результат линейной детекции невалиден)
    3. Геометрический fallback (если предыдущие методы не сработали)
    4. Детекция всплывающих диалогов (добавляется к результату)

    Оптимизация: результаты кэшируются по размеру кадра.
    SCADA-экраны имеют фиксированную раскладку, поэтому зоны не меняются
    между кадрами. Кэш инвалидируется при смене сцены.

    Args:
        frame: Входной кадр BGR (предварительно обработанный).

    Returns:
        Список границ зон с нормализованными координатами.
    """
    t0 = time.perf_counter()
    h, w = frame.shape[:2]

    # Проверяем кэш — зоны SCADA статичны между кадрами
    cache_key = (h, w)
    if cache_key in _zone_cache:
        logger.debug("Зоны из кэша: %d зон для %dx%d", len(_zone_cache[cache_key]), w, h)
        return _zone_cache[cache_key]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === ЭТАП 1: Линейная детекция ===
    zones, boundaries = _segment_zones_line_detection(gray, h, w)

    # === ЭТАП 2: Профили проекций (если линейная детекция невалидна) ===
    if not _is_valid_segmentation(zones, h, w):
        logger.info("Линейная детекция невалидна, пробуем профили проекций")
        zones, boundaries = _segment_zones_projection(gray, h, w)

    # === ЭТАП 3: Геометрический fallback ===
    if not _is_valid_segmentation(zones, h, w):
        logger.info("Профили проекций невалидны, используем геометрический fallback")
        zones, boundaries = _get_geometric_zones(h, w)

    # === ЭТАП 4: Детекция всплывающих диалогов ===
    popup_result = detect_popup_dialog(frame)
    if popup_result.detected:
        logger.debug(
            "Обнаружен всплывающий диалог: bbox=(%.2f, %.2f, %.2f, %.2f)",
            popup_result.bbox.x1 if popup_result.bbox else 0,
            popup_result.bbox.y1 if popup_result.bbox else 0,
            popup_result.bbox.x2 if popup_result.bbox else 0,
            popup_result.bbox.y2 if popup_result.bbox else 0,
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.debug("Сегментация зон завершена: %d зон, %.1f мс", len(zones), elapsed_ms)

    # Кэшируем результат — зоны SCADA статичны между кадрами
    _zone_cache[cache_key] = zones

    return zones


def _segment_zones_line_detection(
    gray: np.ndarray,
    h: int,
    w: int,
) -> tuple[list[ZoneBoundary], _ZoneBoundaries]:
    """Сегментация зон методом линейной детекции (Canny + морфология).

    Это основной метод сегментации, использующий детекцию краёв
    и морфологические операции для поиска разделительных линий.

    Args:
        gray: Полутоновое изображение.
        h: Высота кадра.
        w: Ширина кадра.

    Returns:
        Кортеж (список зон, границы в пикселях).
    """
    # Детектируем вертикальные линии для разделения панелей
    edges = cv2.Canny(gray, 50, 150)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)

    # Ищем горизонтальные разделители
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)

    # Определяем границы зон с использованием edge density analysis
    left_nav_x = _detect_left_nav_boundary(vertical_lines, w)
    right_panel_x = _detect_right_panel_boundary(vertical_lines, w)
    header_y = _detect_header_boundary(horizontal_lines, h)
    bottom_y = _detect_bottom_boundary(horizontal_lines, h)

    # Проверяем, найдены ли значимые разделители
    has_left_divider = _has_significant_vertical_divider(vertical_lines, w, 0, w // 3)
    has_right_divider = _has_significant_vertical_divider(vertical_lines, w, 2 * w // 3, w)
    has_header_divider = _has_significant_horizontal_divider(horizontal_lines, h, 0, h // 10)
    has_bottom_divider = _has_significant_horizontal_divider(horizontal_lines, h, 9 * h // 10, h)

    # Если не найдены разделители — fallback: весь кадр = CENTRAL_SCHEMA
    if not any([has_left_divider, has_right_divider, has_header_divider, has_bottom_divider]):
        logger.debug("Разделители не найдены линейной детекцией")
        return [
            ZoneBoundary(
                zone=ZoneType.CENTRAL_SCHEMA,
                bbox=BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
            )
        ], _ZoneBoundaries(0, w, 0, h)

    # Корректируем границы: если разделитель не найден, используем полный размер кадра
    # для предотвращения создания "фантомных" зон
    if not has_left_divider:
        left_nav_x = 0
    if not has_right_divider:
        right_panel_x = w
    if not has_header_divider:
        header_y = 0
    if not has_bottom_divider:
        bottom_y = h

    # Логируем обнаруженные границы для отладки
    logger.debug(
        "Границы зон (линейная детекция): left_nav=%dpx, right_panel=%dpx, header=%dpx, bottom=%dpx",
        left_nav_x, right_panel_x, header_y, bottom_y
    )

    boundaries = _ZoneBoundaries(
        left_nav_x=left_nav_x,
        right_panel_x=right_panel_x,
        header_y=header_y,
        bottom_y=bottom_y,
    )

    zones = _build_zones_from_boundaries(boundaries, h, w)
    return zones, boundaries


def _segment_zones_projection(
    gray: np.ndarray,
    h: int,
    w: int,
) -> tuple[list[ZoneBoundary], _ZoneBoundaries]:
    """Сегментация зон методом профилей проекций.

    Анализирует горизонтальную и вертикальную плотность пикселей
    для определения границ зон. Используется как промежуточный
    метод между линейной детекцией и геометрическим fallback.

    Алгоритм:
    1. Бинаризация изображения (порог ~200, инверсия)
    2. Вычисление горизонтальной проекции (плотность по столбцам)
    3. Вычисление вертикальной проекции (плотность по строкам)
    4. Нормализация проекций
    5. Поиск "разрывов" где плотность < 5% от максимума
    6. Группировка разрывов в кластеры для определения границ зон

    Args:
        gray: Полутоновое изображение.
        h: Высота кадра.
        w: Ширина кадра.

    Returns:
        Кортеж (список зон, границы в пикселях).
    """
    # Бинаризация: светлые участки становятся 0, тёмные — 255
    # SCADA-интерфейсы обычно имеют тёмный текст на светлом фоне
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Горизонтальная проекция: сумма по столбцам (ось 0)
    # Показывает плотность контента по горизонтали
    h_projection = cv2.reduce(binary, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).flatten()

    # Вертикальная проекция: сумма по строкам (ось 1)
    # Показывает плотность контента по вертикали
    v_projection = cv2.reduce(binary, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F).flatten()

    # Нормализация проекций к диапазону [0, 1]
    h_max = h_projection.max() if h_projection.max() > 0 else 1.0
    v_max = v_projection.max() if v_projection.max() > 0 else 1.0
    h_projection_norm = h_projection / h_max
    v_projection_norm = v_projection / v_max

    # Находим границы зон по разрывам в проекциях
    # Разрыв = область с низкой плотностью (< 5% от максимума)
    gap_threshold = 0.05

    # Границы по горизонтали (left_nav_x, right_panel_x)
    left_nav_x = _find_zone_boundary_from_projection(
        h_projection_norm, w, gap_threshold, search_left=True
    )
    right_panel_x = _find_zone_boundary_from_projection(
        h_projection_norm, w, gap_threshold, search_left=False
    )

    # Границы по вертикали (header_y, bottom_y)
    header_y = _find_zone_boundary_from_projection(
        v_projection_norm, h, gap_threshold, search_left=True
    )
    bottom_y = _find_zone_boundary_from_projection(
        v_projection_norm, h, gap_threshold, search_left=False
    )

    logger.debug(
        "Границы зон (профили проекций): left_nav=%dpx, right_panel=%dpx, header=%dpx, bottom=%dpx",
        left_nav_x, right_panel_x, header_y, bottom_y
    )

    boundaries = _ZoneBoundaries(
        left_nav_x=left_nav_x,
        right_panel_x=right_panel_x,
        header_y=header_y,
        bottom_y=bottom_y,
    )

    zones = _build_zones_from_boundaries(boundaries, h, w)
    return zones, boundaries


def _find_zone_boundary_from_projection(
    projection: np.ndarray,
    length: int,
    gap_threshold: float,
    search_left: bool,
) -> int:
    """Находит границу зоны по разрывам в проекции.

    Ищет кластеры низкоплотных областей для определения
    границ между зонами.

    Args:
        projection: Нормализованная проекция (значения 0-1).
        length: Длина проекции (ширина или высота).
        gap_threshold: Порог для определения разрыва (доля от максимума).
        search_left: True для поиска левой границы, False для правой.

    Returns:
        Координата границы в пикселях.
    """
    # Находим индексы где плотность ниже порога (разрывы)
    gaps = projection < gap_threshold

    # Конвертируем в индексы
    gap_indices = np.where(gaps)[0]

    if len(gap_indices) == 0:
        # Разрывов нет — используем геометрические значения по умолчанию
        if search_left:
            return int(length * 0.12)  # 12% ширины для LEFT_NAV
        else:
            return int(length * 0.78)  # 78% ширины для RIGHT_PANEL

    if search_left:
        # Ищем первый значимый разрыв в левой части (до 30% длины)
        search_region = gap_indices[gap_indices < length * 0.3]
        if len(search_region) > 0:
            # Группируем смежные индексы в кластеры
            clusters = _cluster_indices(search_region)
            if clusters:
                # Берём самый большой кластер в левой части
                largest_cluster = max(clusters, key=len)
                # Граница зоны — правый край кластера
                return int(largest_cluster[-1])
        return int(length * 0.12)
    else:
        # Ищем первый значимый разрыв в правой части (после 70% длины)
        search_region = gap_indices[gap_indices > length * 0.7]
        if len(search_region) > 0:
            # Группируем смежные индексы в кластеры
            clusters = _cluster_indices(search_region)
            if clusters:
                # Берём самый большой кластер в правой части
                largest_cluster = max(clusters, key=len)
                # Граница зоны — левый край кластера
                return int(largest_cluster[0])
        return int(length * 0.78)


def _cluster_indices(indices: np.ndarray) -> list[np.ndarray]:
    """Группирует смежные индексы в кластеры.

    Используется для группировки разрывов в проекции
    в непрерывные области.

    Args:
        indices: Массив индексов для кластеризации.

    Returns:
        Список кластеров (каждый — массив смежных индексов).
    """
    if len(indices) == 0:
        return []

    clusters = []
    current_cluster = [indices[0]]

    for i in range(1, len(indices)):
        # Если индекс смежный (разница <= 2 для учёта шума)
        if indices[i] - indices[i - 1] <= 2:
            current_cluster.append(indices[i])
        else:
            clusters.append(np.array(current_cluster))
            current_cluster = [indices[i]]

    clusters.append(np.array(current_cluster))
    return clusters


def _get_geometric_zones(h: int, w: int) -> tuple[list[ZoneBoundary], _ZoneBoundaries]:
    """Возвращает геометрические зоны по захардкоженным процентам.

    Используется как последний fallback когда линейная детекция
    и профили проекций не сработали. Основано на анализе
    реальных видео SCADA.

    Стандартные проценты для SCADA-раскладок:
    - HEADER: верхние 8% высоты
    - LEFT_NAV: левые 12% ширины
    - RIGHT_PANEL: правые 22% ширины
    - BOTTOM_BAR: нижние 6% высоты
    - CENTRAL_SCHEMA: центральная область (12%-78% ширины, 8%-94% высоты)

    Args:
        h: Высота кадра в пикселях.
        w: Ширина кадра в пикселях.

    Returns:
        Кортеж (список зон, границы в пикселях).
    """
    # Захардкоженные проценты из анализа реальных видео
    header_pct = 0.08      # 8% высоты
    left_nav_pct = 0.12    # 12% ширины
    right_panel_pct = 0.22 # 22% ширины (от правого края)
    bottom_pct = 0.06      # 6% высоты

    # Вычисляем границы в пикселях
    header_y = int(h * header_pct)
    left_nav_x = int(w * left_nav_pct)
    right_panel_x = int(w * (1 - right_panel_pct))
    bottom_y = int(h * (1 - bottom_pct))

    logger.debug(
        "Границы зон (геометрический fallback): left_nav=%dpx, right_panel=%dpx, header=%dpx, bottom=%dpx",
        left_nav_x, right_panel_x, header_y, bottom_y
    )

    boundaries = _ZoneBoundaries(
        left_nav_x=left_nav_x,
        right_panel_x=right_panel_x,
        header_y=header_y,
        bottom_y=bottom_y,
    )

    zones = _build_zones_from_boundaries(boundaries, h, w)
    return zones, boundaries


def _build_zones_from_boundaries(
    boundaries: _ZoneBoundaries,
    h: int,
    w: int,
) -> list[ZoneBoundary]:
    """Строит список зон по границам в пикселях.

    Args:
        boundaries: Границы зон в пикселях.
        h: Высота кадра.
        w: Ширина кадра.

    Returns:
        Список границ зон с нормализованными координатами.
    """
    zones: list[ZoneBoundary] = []

    # Header
    zones.append(
        ZoneBoundary(
            zone=ZoneType.HEADER,
            bbox=BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=boundaries.header_y / h),
        )
    )

    # Left Nav
    zones.append(
        ZoneBoundary(
            zone=ZoneType.LEFT_NAV,
            bbox=BoundingBox(
                x1=0.0,
                y1=boundaries.header_y / h,
                x2=boundaries.left_nav_x / w,
                y2=boundaries.bottom_y / h,
            ),
        )
    )

    # Central Schema (PRIMARY)
    zones.append(
        ZoneBoundary(
            zone=ZoneType.CENTRAL_SCHEMA,
            bbox=BoundingBox(
                x1=boundaries.left_nav_x / w,
                y1=boundaries.header_y / h,
                x2=boundaries.right_panel_x / w,
                y2=boundaries.bottom_y / h,
            ),
        )
    )

    # Right Panel (SECONDARY)
    zones.append(
        ZoneBoundary(
            zone=ZoneType.RIGHT_PANEL,
            bbox=BoundingBox(
                x1=boundaries.right_panel_x / w,
                y1=boundaries.header_y / h,
                x2=1.0,
                y2=boundaries.bottom_y / h,
            ),
        )
    )

    # Bottom Bar
    zones.append(
        ZoneBoundary(
            zone=ZoneType.BOTTOM_BAR,
            bbox=BoundingBox(x1=0.0, y1=boundaries.bottom_y / h, x2=1.0, y2=1.0),
        )
    )

    return zones


def _is_valid_segmentation(zones: list[ZoneBoundary], h: int, w: int) -> bool:
    """Проверяет валидность сегментации.

    Критерии валидности:
    - Найдено ровно 5 зон
    - Центральная зона имеет достаточный размер (>= 30% площади)
    - Все зоны имеют положительную площадь

    Args:
        zones: Список зон для проверки.
        h: Высота кадра.
        w: Ширина кадра.

    Returns:
        True если сегментация валидна.
    """
    if len(zones) != 5:
        return False

    # Находим центральную зону
    central = None
    for zone in zones:
        if zone.zone == ZoneType.CENTRAL_SCHEMA:
            central = zone
            break

    if central is None:
        return False

    # Проверяем минимальный размер центральной зоны
    central_area = (central.bbox.x2 - central.bbox.x1) * (central.bbox.y2 - central.bbox.y1)
    if central_area < 0.30:  # Минимум 30% площади кадра
        logger.debug("Центральная зона слишком мала: %.2f%% площади", central_area * 100)
        return False

    # Проверяем, что все зоны имеют положительную площадь
    for zone in zones:
        area = (zone.bbox.x2 - zone.bbox.x1) * (zone.bbox.y2 - zone.bbox.y1)
        if area <= 0:
            logger.debug("Зона %s имеет нулевую площадь", zone.zone)
            return False

    return True


@dataclass
class PopupDialogResult:
    """Результат детекции всплывающего диалога.

    Атрибуты:
        detected: True если диалог обнаружен.
        bbox: Ограничивающий прямоугольник диалога (нормализованный).
        mask: Маска диалога (опционально).
    """
    detected: bool
    bbox: BoundingBox | None = None
    mask: np.ndarray | None = None


def detect_popup_dialog(frame: np.ndarray) -> PopupDialogResult:
    """Детектирует всплывающие диалоги поверх центральной зоны.

    Ищет большие прямоугольные области с отчётливыми границами,
    накладываемые на центральную зону мнемосхемы.

    Критерии диалога:
    - Большая прямоугольная область (> 15% площади кадра)
    - Светлый или бело-серый фон
    - Высокая плотность краёв по периметру (рамка диалога)
    - Контраст с окружающим фоном

    Args:
        frame: Входной кадр BGR.

    Returns:
        PopupDialogResult с результатами детекции.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ищем светлые прямоугольные области
    # Диалоги обычно светлее основного фона SCADA
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Морфологическая обработка для удаления шума
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Находим контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = h * w
    min_dialog_area = frame_area * 0.15  # Минимум 15% площади кадра
    max_dialog_area = frame_area * 0.80  # Максимум 80% (иначе это не диалог)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_dialog_area or area > max_dialog_area:
            continue

        # Аппроксимируем контур прямоугольником
        x, y, bw, bh = cv2.boundingRect(contour)
        rect_area = bw * bh

        # Проверяем, что контур действительно прямоугольный
        # (отношение площади контура к площади bounding rect должно быть высоким)
        if rect_area == 0:
            continue
        rectangularity = area / rect_area
        if rectangularity < 0.8:  # Должен быть достаточно прямоугольным
            continue

        # Проверяем плотность краёв по периметру (рамка диалога)
        edge_density = _compute_border_edge_density(gray, x, y, bw, bh)
        if edge_density < 0.1:  # Минимум 10% краёв по периметру
            continue

        # Найден диалог
        bbox = BoundingBox(
            x1=x / w,
            y1=y / h,
            x2=(x + bw) / w,
            y2=(y + bh) / h,
        )

        # Создаём маску диалога
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)

        logger.debug(
            "Детектирован диалог: bbox=(%d, %d, %d, %d), площадь=%.1f%%, края=%.1f%%",
            x, y, x + bw, y + bh, area / frame_area * 100, edge_density * 100
        )

        return PopupDialogResult(detected=True, bbox=bbox, mask=mask)

    return PopupDialogResult(detected=False)


def _compute_border_edge_density(
    gray: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> float:
    """Вычисляет плотность краёв по периметру прямоугольника.

    Используется для проверки наличия рамки диалога.

    Args:
        gray: Полутоновое изображение.
        x, y: Координаты верхнего левого угла.
        w, h: Ширина и высота прямоугольника.

    Returns:
        Плотность краёв по периметру (0.0 - 1.0).
    """
    # Детектируем края
    edges = cv2.Canny(gray, 50, 150)

    # Извлекаем периметр (с толщиной границы)
    border_thickness = max(5, min(w, h) // 20)

    # Создаём маску периметра
    mask = np.zeros_like(edges)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, border_thickness)
    # Убираем внутреннюю часть
    inner_margin = border_thickness
    if w > 2 * inner_margin and h > 2 * inner_margin:
        cv2.rectangle(
            mask,
            (x + inner_margin, y + inner_margin),
            (x + w - inner_margin, y + h - inner_margin),
            0, -1
        )

    # Считаем краевые пиксели на периметре
    border_pixels = np.sum(mask > 0)
    if border_pixels == 0:
        return 0.0

    edge_pixels = np.sum((edges > 0) & (mask > 0))
    return edge_pixels / border_pixels


def extract_zone(
    frame: np.ndarray,
    zone_bbox: BoundingBox,
) -> np.ndarray:
    """Извлекает область зоны из кадра.

    Args:
        frame: Входной кадр BGR.
        zone_bbox: Ограничивающий прямоугольник зоны.

    Returns:
        Обрезанный кадр, содержащий только зону.
    """
    h, w = frame.shape[:2]
    x1 = int(zone_bbox.x1 * w)
    y1 = int(zone_bbox.y1 * h)
    x2 = int(zone_bbox.x2 * w)
    y2 = int(zone_bbox.y2 * h)

    # Ограничиваем координаты
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    return frame[y1:y2, x1:x2]


def _detect_left_nav_boundary(vertical_lines: np.ndarray, width: int) -> int:
    """Определяет границу левой навигационной панели.

    Ищет вертикальную линию-разделитель в левой трети экрана.

    Args:
        vertical_lines: Маска вертикальных линий.
        width: Ширина кадра.

    Returns:
        X-координата правой границы левой панели.
    """
    # Сканируем левую треть кадра
    left_third = vertical_lines[:, : width // 3]

    # Суммируем пиксели по вертикали для каждого столбца
    col_sums = left_third.sum(axis=0)

    # Ищем максимум — это вертикальная разделительная линия
    if col_sums.max() > 0:
        # Ищем правый край левой панели
        threshold = col_sums.max() * 0.3
        candidates = np.where(col_sums > threshold)[0]
        if len(candidates) > 0:
            return int(candidates[-1])

    # По умолчанию: 10% ширины кадра
    return width // 10


def _has_significant_vertical_divider(
    vertical_lines: np.ndarray, width: int, x_start: int, x_end: int, min_strength: float = 0.1
) -> bool:
    """Проверяет наличие значимой вертикальной разделительной линии.

    Args:
        vertical_lines: Маска вертикальных линий (значения 0 или 255 от Canny).
        width: Ширина кадра.
        x_start: Начало диапазона по X.
        x_end: Конец диапазона по X.
        min_strength: Минимальная относительная сила линии (0-1).

    Returns:
        True если найдена значимая разделительная линия.

    Note:
        Canny edge пиксели имеют значение 255, поэтому нормализуем сумму
        делением на 255 для получения количества пикселей.
    """
    region = vertical_lines[:, x_start:x_end]
    if region.size == 0:
        return False
    col_sums = region.sum(axis=0)
    # Нормализуем: Canny пиксели = 255, делим для получения количества пикселей
    col_sums_normalized = col_sums / 255.0
    threshold = vertical_lines.shape[0] * min_strength
    return (col_sums_normalized > threshold).any()


def _has_significant_horizontal_divider(
    horizontal_lines: np.ndarray, height: int, y_start: int, y_end: int, min_strength: float = 0.05
) -> bool:
    """Проверяет наличие значимой горизонтальной разделительной линии.

    Args:
        horizontal_lines: Маска горизонтальных линий (значения 0 или 255 от Canny).
        height: Высота кадра.
        y_start: Начало диапазона по Y.
        y_end: Конец диапазона по Y.
        min_strength: Минимальная относительная сила линии (0-1).

    Returns:
        True если найдена значимая разделительная линия.

    Note:
        Canny edge пиксели имеют значение 255, поэтому нормализуем сумму
        делением на 255 для получения количества пикселей.
    """
    region = horizontal_lines[y_start:y_end, :]
    if region.size == 0:
        return False
    row_sums = region.sum(axis=1)
    # Нормализуем: Canny пиксели = 255, делим для получения количества пикселей
    row_sums_normalized = row_sums / 255.0
    threshold = horizontal_lines.shape[1] * min_strength
    return (row_sums_normalized > threshold).any()


def _detect_right_panel_boundary(vertical_lines: np.ndarray, width: int) -> int:
    """Определяет границу правой панели деталей.

    Args:
        vertical_lines: Маска вертикальных линий.
        width: Ширина кадра.

    Returns:
        X-координату левой границы правой панели.
    """
    # Сканируем правую треть кадра
    right_start = 2 * width // 3
    right_third = vertical_lines[:, right_start:]

    col_sums = right_third.sum(axis=0)

    if col_sums.max() > 0:
        threshold = col_sums.max() * 0.3
        candidates = np.where(col_sums > threshold)[0]
        if len(candidates) > 0:
            return right_start + int(candidates[0])

    # По умолчанию: 80% ширины кадра
    return int(width * 0.8)


def _detect_header_boundary(horizontal_lines: np.ndarray, height: int) -> int:
    """Определяет нижнюю границу заголовка.

    Args:
        horizontal_lines: Маска горизонтальных линий.
        height: Высота кадра.

    Returns:
        Y-координату нижней границы заголовка.
    """
    # Сканируем верхнюю десятую часть
    top_tenth = horizontal_lines[: height // 10, :]
    row_sums = top_tenth.sum(axis=1)

    if row_sums.max() > 0:
        threshold = row_sums.max() * 0.3
        candidates = np.where(row_sums > threshold)[0]
        if len(candidates) > 0:
            return int(candidates[-1])

    # По умолчанию: 6% высоты
    return height // 16


def _detect_bottom_boundary(horizontal_lines: np.ndarray, height: int) -> int:
    """Определяет верхнюю границу нижней строки состояния.

    Args:
        horizontal_lines: Маска горизонтальных линий.
        height: Высота кадра.

    Returns:
        Y-координату верхней границы нижней панели.
    """
    # Сканируем нижнюю десятую часть
    bottom_start = 9 * height // 10
    bottom_tenth = horizontal_lines[bottom_start:, :]
    row_sums = bottom_tenth.sum(axis=1)

    if row_sums.max() > 0:
        threshold = row_sums.max() * 0.3
        candidates = np.where(row_sums > threshold)[0]
        if len(candidates) > 0:
            return bottom_start + int(candidates[0])

    # По умолчанию: 95% высоты
    return int(height * 0.95)
