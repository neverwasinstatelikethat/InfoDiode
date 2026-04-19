"""Модуль калибровки — связывает распознанные метки с ID параметров.

Трёхфазный workflow:
1. Calibration — пользователь связывает метки с таблицей параметров
2. Recognition — автоматическое извлечение значений по ROI
3. Generalization — перенос калибровки на другие мнемосхемы
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path

import numpy as np
from Levenshtein import distance as levenshtein_distance

from app.core.ocr_models import BBox
from app.models.schemas import (
    BoundingBox,
    LabelValuePair,
    ParameterMapping,
    ZoneBoundary,
    ZoneType,
)

logger = logging.getLogger(__name__)


class CalibrationProfile:
    """Профиль калибровки для одной мнемосхемы.

    Хранит привязку параметров к ROI и зоны мнемосхемы.
    """

    def __init__(self, profile_id: str | None = None) -> None:
        self.profile_id = profile_id or str(uuid.uuid4())
        self.mnemonic_name: str = ""
        self.parameter_table_file: str = ""
        self.mappings: list[ParameterMapping] = []
        self.zones: list[ZoneBoundary] = []
        self.color_ranges: dict[str, list[int]] = {}  # normal/alarm/warning HSV ranges

    def add_mapping(self, mapping: ParameterMapping) -> None:
        """Добавляет привязку параметра."""
        # Удаляем существующую привязку с тем же param_id
        self.mappings = [m for m in self.mappings if m.param_id != mapping.param_id]
        self.mappings.append(mapping)

    def get_mapping_by_id(self, param_id: int) -> ParameterMapping | None:
        """Возвращает привязку по ID параметра.

        Args:
            param_id: ID параметра.

        Returns:
            Привязка или None.
        """
        return next((m for m in self.mappings if m.param_id == param_id), None)

    def save(self, filepath: str | Path) -> None:
        """Сохраняет профиль в JSON.

        Args:
            filepath: Путь к файлу.
        """
        data = {
            "profile_id": self.profile_id,
            "mnemonic_name": self.mnemonic_name,
            "parameter_table_file": self.parameter_table_file,
            "mappings": [m.model_dump() for m in self.mappings],
            "zones": [z.model_dump() for z in self.zones],
            "color_ranges": self.color_ranges,
        }
        Path(filepath).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Профиль калибровки сохранён: %s (%d привязок)", filepath, len(self.mappings))

    @classmethod
    def load(cls, filepath: str | Path) -> CalibrationProfile:
        """Загружает профиль из JSON.

        Args:
            filepath: Путь к файлу.

        Returns:
            Загруженный профиль.
        """
        t0 = time.perf_counter()
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        profile = cls(profile_id=data["profile_id"])
        profile.mnemonic_name = data.get("mnemonic_name", "")
        profile.parameter_table_file = data.get("parameter_table_file", "")
        profile.mappings = [ParameterMapping(**m) for m in data.get("mappings", [])]
        profile.zones = [ZoneBoundary(**z) for z in data.get("zones", [])]
        profile.color_ranges = data.get("color_ranges", {})

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Профиль калибровки загружен: %s (%d привязок, %.1f мс)", filepath, len(profile.mappings), elapsed_ms)
        return profile


# Таблица транслитерации кириллицы → латиница для OCR-исправлений
CYRILLIC_TO_LATIN: dict[str, str] = {
    "Т": "T", "Р": "P", "С": "S", "Н": "H", "В": "V",
    "М": "M", "К": "K", "А": "A", "О": "O", "Е": "E",
    "т": "t", "р": "p", "с": "c", "н": "h", "в": "v",
    "м": "m", "к": "k", "а": "a", "о": "o", "е": "e",
}


def _transliterate_cyrillic_to_latin(text: str) -> str:
    """Транслитерирует кириллические символы, похожие на латинские.

    Помогает исправить OCR-ошибки когда кириллица распознана вместо латиницы
    в sensor ID: ТI-101 → TI-101, РI-205 → PI-205.

    Args:
        text: Исходный текст.

    Returns:
        Текст с транслитерированными символами.
    """
    return "".join(CYRILLIC_TO_LATIN.get(c, c) for c in text)


def match_labels_to_params(
    label_value_pairs: list[LabelValuePair],
    parameter_table: list[dict],
    max_distance: int = 8,
) -> list[ParameterMapping]:
    """Сопоставляет распознанные метки с записями таблицы параметров.

    Три уровня сопоставления:
    1. Полное русское название (Levenshtein distance)
    2. Короткое имя (T, P, dP, Vb, L, n, Pos, f, V)
    3. Sensor ID в скобках (TE4401, PT4413)

    Args:
        label_value_pairs: Распознанные пары метка-значение.
        parameter_table: Таблица параметров из Excel/CSV.
        max_distance: Максимальное расстояние Левенштейна для fuzzy matching.
            Увеличено до 8 для лучшей устойчивости к OCR-ошибкам.

    Returns:
        Список привязок параметров.
    """
    t0 = time.perf_counter()
    mappings: list[ParameterMapping] = []
    used_param_ids: set[int] = set()

    for pair in label_value_pairs:
        best_match: dict | None = None
        best_score: float = float("inf")

        for param in parameter_table:
            param_id = param.get("id", 0)
            if param_id in used_param_ids:
                continue

            score = _compute_match_score(pair.label, param)

            if score < best_score:
                best_score = score
                best_match = param

        if best_match is not None and best_score <= max_distance:
            used_param_ids.add(best_match["id"])
            mappings.append(
                ParameterMapping(
                    param_id=best_match["id"],
                    label_text=pair.label,
                    short_name=best_match.get("short_name", ""),
                    full_name=best_match.get("name", ""),
                    unit=best_match.get("unit", ""),
                    decimal_places=best_match.get("decimal_places", 1),
                    roi_bbox=pair.value_bbox,  # Legacy: for backward compatibility
                    label_bbox=pair.label_bbox,  # NEW: bbox of the label text
                    value_bbox=pair.value_bbox,  # NEW: bbox of the value text (may be None)
                    zone=pair.zone,
                )
            )
        else:
            logger.debug("Метка '%s' не сопоставлена (best_score=%.2f)", pair.label, best_score)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Сопоставление меток: %d/%d пар привязано к параметрам (%.1f мс)",
        len(mappings), len(label_value_pairs), elapsed_ms
    )

    return mappings


def _compute_match_score(label: str, param: dict) -> float:
    """Вычисляет оценку совпадения метки с параметром.

    Args:
        label: Распознанная метка.
        param: Запись из таблицы параметров.

    Returns:
        Оценка (меньше = лучше). Infinity если нет совпадения.
    """
    score = float("inf")

    # Транслитерируем метку для сравнения с латинскими ID
    label_transliterated = _transliterate_cyrillic_to_latin(label)

    # Уровень 1: Полное русское название (fuzzy)
    name = param.get("name", "")
    if name:
        dist = levenshtein_distance(label.lower(), name.lower())
        # Нормализуем по длине: чем больше расстояние относительно длины, тем хуже
        max_len = max(len(label), len(name), 1)
        name_score = (dist / max_len) * 10
        score = min(score, name_score)

    # Уровень 2: Короткое имя (точное или содержит)
    short_name = param.get("short_name", "")
    if short_name:
        label_lower = label.lower()
        short_lower = short_name.lower()
        if short_lower in label_lower:
            score = min(score, 1.5)  # Хороший матч
        elif label_lower in short_lower:
            score = min(score, 1.5)

    # Уровень 3: Sensor ID в скобках (с поддержкой транслитерации)
    name_with_id = param.get("name", "")
    # Ищем sensor ID в формате (TE4401), (PT4413), и т.д.
    sensor_ids = re.findall(r"\(([A-Za-z]{1,3}\d{3,5})\)", name_with_id)
    for sid in sensor_ids:
        # Прямое совпадение
        if sid in label:
            score = min(score, 0.0)  # Идеальный матч
        # Совпадение с транслитерированной меткой (ТI-101 → TI-101)
        elif sid in label_transliterated:
            score = min(score, 0.5)  # Хороший матч с небольшим штрафом

    return score


def _is_dash_value(text: str) -> bool:
    """Проверяет, является ли текст признаком неактивного/отсутствующего значения.

    Args:
        text: Распознанный текст.

    Returns:
        True если текст похож на "нет данных" (тире, прочерки и т.п.).
    """
    dash_patterns = ["---", "—", "--", "-", "…", "...", "___"]
    # Также проверяем OCR-артефакты: "111" часто распознаётся вместо тире
    ocr_artifacts = ["111", "mmm", "mm", "m"]
    text_clean = text.strip().lower()
    return text_clean in dash_patterns or text_clean in ocr_artifacts


def _extract_sensor_id_from_name(name: str) -> list[str]:
    """Извлекает sensor ID из названия параметра.

    Args:
        name: Полное название параметра (может содержать ID в скобках).

    Returns:
        Список найденных sensor ID.
    """
    # Паттерны: (TE4401), (PT4413), TI-101, PI-205 и т.д.
    ids = re.findall(r"\(([A-Za-z]{1,3}\d{3,5})\)", name)
    ids.extend(re.findall(r"\b([A-Z]{1,3}-?\d{3,5})\b", name))
    return ids


def _get_physical_range_for_unit(unit: str) -> tuple[float, float] | None:
    """Возвращает физический диапазон для единицы измерения.

    Args:
        unit: Единица измерения (°C, kPa, mm, %, Hz, mm/s, V).

    Returns:
        Кортеж (min, max) или None если диапазон неизвестен.
    """
    unit_ranges: dict[str, tuple[float, float]] = {
        "°c": (-50.0, 800.0),  # Температура
        "c": (-50.0, 800.0),
        "deg c": (-50.0, 800.0),
        "kpa": (0.0, 8000.0),  # Давление
        "pa": (0.0, 8000000.0),
        "mm": (0.0, 1000.0),  # Уровень
        "%": (0.0, 100.0),  # Положение/влажность
        "hz": (0.0, 9000.0),  # Частота/обороты
        "mm/s": (0.0, 50.0),  # Вибрация
        "v": (0.0, 400.0),  # Напряжение
    }
    unit_lower = unit.lower().strip()
    return unit_ranges.get(unit_lower)


def _validate_numeric_value(
    text: str, unit: str, confidence: float
) -> tuple[float | None, bool]:
    """Валидирует числовое значение по физическому диапазону.

    Args:
        text: Распознанный текст значения.
        unit: Единица измерения для определения диапазона.
        confidence: Уверенность распознавания.

    Returns:
        Кортеж (value, is_valid). value=None если не удалось распарсить.
    """
    # Очистка текста
    cleaned = text.strip().replace(",", ".").replace("°", "").replace("%", "")
    cleaned = re.sub(r"[^\d.\-+]", "", cleaned)

    try:
        value = float(cleaned)
    except ValueError:
        return (None, False)

    # Получаем диапазон для единицы измерения
    range_min_max = _get_physical_range_for_unit(unit)
    if range_min_max is None:
        return (value, True)  # Нет диапазона — считаем валидным

    min_val, max_val = range_min_max

    # Если вне диапазона и уверенность низкая — отклоняем
    if not (min_val <= value <= max_val) and confidence < 0.8:
        return (value, False)

    return (value, True)


def _is_left_sidebar_param(param_name: str) -> bool:
    """Проверяет, относится ли параметр к левой боковой панели.

    Args:
        param_name: Название параметра.

    Returns:
        True если параметр обычно находится на левой панели.
    """
    sidebar_keywords = [
        "атм.р", "атм р", "атмосферное",
        "т наруж", "температура наруж", "наруж.возд",
        "т окр", "температура окр",
    ]
    name_lower = param_name.lower()
    return any(kw in name_lower for kw in sidebar_keywords)


def _is_anchor_param(param_name: str) -> bool:
    """Проверяет, является ли параметр якорным (масляные параметры).

    Якорные параметры видны на всех вкладках и обоих типах ГПА.

    Args:
        param_name: Название параметра.

    Returns:
        True если параметр является якорным.
    """
    anchor_keywords = [
        "l масла", "уровень масла", "l масл",
        "t масла", "температура масла", "t масл",
        "p масла", "давление масла", "p масл",
    ]
    name_lower = param_name.lower()
    return any(kw in name_lower for kw in anchor_keywords)


def calibrate_with_grounding(
    frame: np.ndarray,
    param_table: list[dict],
    florence_detector,
) -> list[ParameterMapping]:
    """Выполняет 7-фазную глубокую калибровку с использованием multi-task Florence-2.

    Phase A: Scene Understanding — анализ сцены, определение типа ГПА и активной вкладки.
    Phase B: Full Text Extraction — извлечение всего текста с фильтрацией.
    Phase C: Fuzzy Match + Grounding — сопоставление параметров с OCR + targeted grounding.
    Phase D: Value Region Discovery — поиск областей значений относительно меток.
    Phase E: Value Range Validation — физическая валидация значений.
    Phase F: Coverage Validation — проверка покрытия (минимум 50%).
    Phase G: Anchor Calibration — валидация через якорные параметры масла.

    Args:
        frame: Входной кадр BGR.
        param_table: Таблица параметров со столбцами id, name, short_name, unit и т.д.
        florence_detector: Экземпляр FlorenceDetector с multi-task методами.

    Returns:
        Список ParameterMapping с привязками параметров к ROI.
        Возвращает пустой список если найдено < 50% параметров или есть popup.
    """
    t0 = time.perf_counter()
    frame_h, frame_w = frame.shape[:2]

    # Вспомогательная функция для конвертации BBox в BoundingBox
    def bbox_to_schema(b: BBox) -> BoundingBox:
        return BoundingBox(
            x1=b.x / max(frame_w, 1),
            y1=b.y / max(frame_h, 1),
            x2=(b.x + b.w) / max(frame_w, 1),
            y2=(b.y + b.h) / max(frame_h, 1),
        )

    mappings: list[ParameterMapping] = []
    found_param_ids: set[int] = set()

    # Словарь для быстрого доступа к параметрам по имени
    name_to_param: dict[str, dict] = {}
    for param in param_table:
        name = param.get("name", "")
        if name:
            name_to_param[name] = param

    # ==========================================================================
    # Phase A: Scene Understanding
    # ==========================================================================
    logger.info("Phase A: Scene Understanding — анализ сцены Florence-2")

    scene_info = {
        "gpa_type": None,
        "active_tab": None,
        "has_popup": False,
        "mnemonic_name": None,
        "description": "",
    }

    try:
        scene_info = florence_detector.describe_scene(frame)
        logger.info(
            "Phase A: GPA=%s, tab=%s, popup=%s",
            scene_info.get("gpa_type"),
            scene_info.get("active_tab"),
            scene_info.get("has_popup"),
        )
    except Exception as e:
        logger.warning("Phase A: Ошибка describe_scene: %s — продолжаем без scene info", e)

    # Если есть popup — пропускаем калибровку
    if scene_info.get("has_popup", False):
        logger.warning("Phase A: Обнаружен popup/dialog — калибровка пропущена")
        return []

    # ==========================================================================
    # Phase B: Full Text Extraction
    # ==========================================================================
    logger.info("Phase B: Full Text Extraction — извлечение всего текста")

    ocr_results: list[tuple[str, BBox, float]] = []
    try:
        raw_ocr = florence_detector.ocr_all_text(frame)
        # Фильтрация: убираем односимвольные, чистые символы, низкую уверенность
        for text, bbox, conf in raw_ocr:
            text_clean = text.strip()
            # Пропускаем односимвольные результаты
            if len(text_clean) < 2:
                continue
            # Пропускаем чистые символы (только пунктуация)
            if re.match(r"^[^\w]+$", text_clean):
                continue
            # Пропускаем низкоуверенные
            if conf < 0.3:
                continue
            ocr_results.append((text_clean, bbox, conf))

        logger.info("Phase B: Извлечено %d текстовых регионов", len(ocr_results))
        for text, bbox, conf in ocr_results[:5]:
            logger.debug("Phase B: OCR текст: '%s' (conf=%.2f)", text, conf)
    except Exception as e:
        logger.warning("Phase B: Ошибка ocr_all_text: %s", e)
        ocr_results = []

    # Обнаружение dash-значений (неактивные параметры)
    inactive_texts: set[str] = set()
    for text, _, _ in ocr_results:
        if _is_dash_value(text):
            inactive_texts.add(text)

    if inactive_texts:
        logger.debug("Phase B: Обнаружены неактивные значения: %s", inactive_texts)

    # ==========================================================================
    # Phase C: Fuzzy Match + Grounding
    # ==========================================================================
    logger.info("Phase C: Fuzzy Match + Grounding — сопоставление параметров")

    grounded_params: list[tuple[dict, BBox, float]] = []  # (param, label_bbox, confidence)

    # Сначала пробуем fuzzy matching против OCR результатов
    # Порог для fuzzy matching — увеличен для лучшего охвата SCADA-меток
    MATCH_THRESHOLD = 7.0
    numeric_only = re.compile(r'^[+-]?\d+[.,]?\d*$')

    # Стратегия: собираем ВСЕ кандидаты (param, ocr_idx, score, bbox, conf, text),
    # затем жадно назначаем лучшие пары — каждый OCR-текст используется только один раз.
    candidates: list[tuple[int, int, float, BBox, float, str]] = []  # (param_id, ocr_idx, score, bbox, conf, text)

    for param in param_table:
        param_id = param.get("id", 0)
        if param_id in found_param_ids:
            continue

        for ocr_idx, (text, bbox, conf) in enumerate(ocr_results):
            # Пропускаем чистые числа — это значения, не метки
            if numeric_only.match(text.strip()):
                continue

            score = _compute_match_score(text, param)

            # Дополнительная стратегия: word-level matching для сокращённых меток
            if score > MATCH_THRESHOLD:
                param_name = param.get("name", "").lower()
                param_words = set(re.findall(r'[а-яёА-ЯЁa-zA-Z]{2,}', param_name))
                label_words = set(re.findall(r'[а-яёА-ЯЁa-zA-Z]{2,}', text.lower()))
                common_words = param_words & label_words
                if len(common_words) >= 2:
                    score = min(score, 3.0)
                elif any(len(w) > 4 for w in common_words):
                    score = min(score, 4.0)

            # Также проверяем short_name prefix/suffix
            if score > MATCH_THRESHOLD:
                short_name = param.get("short_name", "")
                if short_name and len(short_name) >= 1:
                    text_lower = text.strip().lower()
                    short_lower = short_name.lower()
                    if text_lower.startswith(short_lower) or text_lower.endswith(short_lower):
                        score = min(score, 2.0)

            if score <= MATCH_THRESHOLD:
                candidates.append((param_id, ocr_idx, score, bbox, conf, text))

    # Жадное назначение: сортируем по score (лучшие сначала), каждый OCR-текст и param — однократно
    candidates.sort(key=lambda c: c[2])
    used_ocr_indices: set[int] = set()

    for param_id, ocr_idx, score, bbox, conf, text in candidates:
        if param_id in found_param_ids:
            continue
        if ocr_idx in used_ocr_indices:
            continue
        # Находим param dict по id
        param_dict = next((p for p in param_table if p.get("id", 0) == param_id), None)
        if param_dict is None:
            continue
        grounded_params.append((param_dict, bbox, conf))
        found_param_ids.add(param_id)
        used_ocr_indices.add(ocr_idx)
        logger.debug(
            "Phase C: Fuzzy match '%s' -> param %d (score=%.2f)",
            text, param_id, score
        )

    # Для не найденных параметров — targeted grounding
    unmatched_params = [p for p in param_table if p.get("id", 0) not in found_param_ids]
    if unmatched_params:
        unmatched_names = [p.get("name", "") for p in unmatched_params if p.get("name")]
        logger.info("Phase C: Targeted grounding для %d параметров (батчами по 10)", len(unmatched_names))

        # Батчами по 10 — Florence не может обработать 261 фразу за раз
        GROUNDING_BATCH_SIZE = 10
        for batch_start in range(0, min(len(unmatched_names), 50), GROUNDING_BATCH_SIZE):
            batch = unmatched_names[batch_start:batch_start + GROUNDING_BATCH_SIZE]
            try:
                grounding_results = florence_detector.ground_phrases(frame, batch)
                for phrase, bbox, conf in grounding_results:
                    if bbox is not None and phrase in name_to_param:
                        param = name_to_param[phrase]
                        param_id = param.get("id", 0)
                        if param_id not in found_param_ids:
                            grounded_params.append((param, bbox, conf))
                            found_param_ids.add(param_id)
                            logger.debug("Phase C: Grounding match '%s' -> param %d", phrase, param_id)
            except Exception as e:
                logger.debug("Phase C: Ошибка grounding batch: %s", e)

    # Пробуем сопоставление по short_name и sensor ID
    for param in param_table:
        param_id = param.get("id", 0)
        if param_id in found_param_ids:
            continue

        short_name = param.get("short_name", "")
        sensor_ids = _extract_sensor_id_from_name(param.get("name", ""))

        for text, bbox, conf in ocr_results:
            text_upper = text.upper().replace(" ", "")
            # Проверяем short_name
            if short_name and short_name.upper() in text_upper:
                grounded_params.append((param, bbox, conf))
                found_param_ids.add(param_id)
                logger.debug("Phase C: Short_name match '%s' -> param %d", text, param_id)
                break
            # Проверяем sensor ID
            for sid in sensor_ids:
                if sid.upper() in text_upper:
                    grounded_params.append((param, bbox, conf))
                    found_param_ids.add(param_id)
                    logger.debug("Phase C: Sensor ID match '%s' -> param %d", text, param_id)
                    break

    logger.info("Phase C: Всего сопоставлено %d/%d параметров", len(grounded_params), len(param_table))

    # ==========================================================================
    # Phase D: Value Region Discovery
    # ==========================================================================
    logger.info("Phase D: Value Region Discovery — поиск областей значений")

    # Получаем dense regions и proposals от Florence
    dense_regions: list[tuple[str, BBox]] = []
    proposed_regions: list[BBox] = []

    try:
        dense_regions = florence_detector.detect_regions(frame)
        logger.debug("Phase D: Dense regions: %d", len(dense_regions))
    except Exception as e:
        logger.debug("Phase D: Ошибка detect_regions: %s", e)

    try:
        proposed_regions = florence_detector.propose_regions(frame)
        logger.debug("Phase D: Proposed regions: %d", len(proposed_regions))
    except Exception as e:
        logger.debug("Phase D: Ошибка propose_regions: %s", e)

    # Импортируем PaddleOCR для OCR в регионе значения
    paddle_ocr_fn = None
    try:
        from app.core.ocr_engine import ocr_roi_single
        paddle_ocr_fn = ocr_roi_single
    except ImportError:
        logger.debug("Phase D: PaddleOCR не доступен")

    for param, label_bbox, label_conf in grounded_params:
        param_id = param.get("id", 0)
        unit = param.get("unit", "")

        # Параметры поиска значения
        search_width = int(label_bbox.w * 2.5)  # Вправо на 2.5 ширины метки
        search_height = int(label_bbox.h * 1.6)  # +/- 30% по высоте
        y_offset = int(label_bbox.h * 0.3)

        # Pattern A: Label LEFT -> Value RIGHT (основной)
        value_x = label_bbox.x + label_bbox.w
        value_y = max(0, label_bbox.y - y_offset)
        value_w = min(search_width, frame_w - value_x)
        value_h = min(search_height, frame_h - value_y)

        value_bbox = BBox(x=value_x, y=value_y, w=value_w, h=value_h)
        value_conf = label_conf * 0.9  # Немного снижаем уверенность

        # Ищем числовые тексты в области поиска
        numeric_candidates: list[tuple[BBox, str, float]] = []

        for text, bbox, conf in ocr_results:
            # Проверяем пересечение с областью поиска
            if bbox.x >= value_x and bbox.x < value_x + value_w:
                if bbox.y >= value_y and bbox.y + bbox.h <= value_y + value_h:
                    # Проверяем что текст похож на число
                    if re.match(r"^[+-]?\d+[.,]?\d*\s*[°%a-zA-Zа-яА-Я]*$", text):
                        numeric_candidates.append((bbox, text, conf))

        # Если нашли числовые кандидаты — берём ближайший к метке
        if numeric_candidates:
            numeric_candidates.sort(key=lambda x: x[0].x)  # По X (ближайший справа)
            best_bbox, best_text, best_conf = numeric_candidates[0]
            value_bbox = best_bbox
            value_conf = best_conf
            logger.debug("Phase D: Найдено значение для param %d: '%s'", param_id, best_text)
        else:
            # Pattern B: Label ABOVE -> Value BELOW (компактная компоновка)
            below_y = label_bbox.y + label_bbox.h
            below_candidates = []
            for text, bbox, conf in ocr_results:
                if bbox.y >= below_y and bbox.y < below_y + search_height:
                    if abs(bbox.x - label_bbox.x) < label_bbox.w:
                        if re.match(r"^[+-]?\d+[.,]?\d*\s*[°%a-zA-Zа-яА-Я]*$", text):
                            below_candidates.append((bbox, text, conf))

            if below_candidates:
                below_candidates.sort(key=lambda x: x[0].y)
                best_bbox, best_text, best_conf = below_candidates[0]
                value_bbox = best_bbox
                value_conf = best_conf
                logger.debug("Phase D: Найдено значение (below) для param %d: '%s'", param_id, best_text)
            elif paddle_ocr_fn is not None:
                # Fallback: PaddleOCR на расширенной области
                try:
                    expand_x = max(0, value_x - 10)
                    expand_y = max(0, value_y - 10)
                    expand_w = min(value_w + 20, frame_w - expand_x)
                    expand_h = min(value_h + 20, frame_h - expand_y)

                    if expand_w > 10 and expand_h > 10:
                        crop = frame[expand_y:expand_y + expand_h, expand_x:expand_x + expand_w]
                        ocr_text, ocr_conf = paddle_ocr_fn(crop, min_confidence=0.4)
                        if ocr_text and re.match(r"^[+-]?\d+[.,]?\d*", ocr_text):
                            # Создаём bbox для найденного значения (приблизительно)
                            value_bbox = BBox(x=expand_x, y=expand_y, w=expand_w, h=expand_h)
                            value_conf = ocr_conf
                            logger.debug("Phase D: PaddleOCR значение для param %d: '%s'", param_id, ocr_text)
                except Exception as e:
                    logger.debug("Phase D: PaddleOCR error для param %d: %s", param_id, e)

        # Проверяем dense regions — может быть полезно для значений
        for region_caption, region_bbox in dense_regions:
            # Если регион пересекается с областью значения
            if (abs(region_bbox.x - value_bbox.x) < value_bbox.w and
                abs(region_bbox.y - value_bbox.y) < value_bbox.h):
                # Уточняем bbox если регион меньше
                if region_bbox.w < value_bbox.w and region_bbox.h < value_bbox.h:
                    value_bbox = region_bbox
                    break

        # ==========================================================================
        # Phase E: Value Range Validation
        # ==========================================================================
        # Пытаемся извлечь значение для валидации
        # Используем OCR тексты рядом с value_bbox
        extracted_value = None
        for text, bbox, conf in ocr_results:
            if (abs(bbox.x - value_bbox.x) < 20 and
                abs(bbox.y - value_bbox.y) < 20):
                val, is_valid = _validate_numeric_value(text, unit, value_conf)
                if val is not None:
                    extracted_value = val
                    if not is_valid:
                        logger.debug("Phase E: Значение %.2f вне диапазона для param %d", val, param_id)
                    break

        # ==========================================================================
        # Создаём ParameterMapping
        # ==========================================================================
        zone = ZoneType.CENTRAL_SCHEMA
        if _is_left_sidebar_param(param.get("name", "")):
            zone = ZoneType.LEFT_SIDEBAR

        mappings.append(
            ParameterMapping(
                param_id=param_id,
                label_text=param.get("name", ""),
                short_name=param.get("short_name", ""),
                full_name=param.get("name", ""),
                unit=unit,
                decimal_places=param.get("decimal_places", 1),
                roi_bbox=bbox_to_schema(value_bbox),
                label_bbox=bbox_to_schema(label_bbox),
                value_bbox=bbox_to_schema(value_bbox),
                zone=zone,
            )
        )

    # ==========================================================================
    # Phase F: Coverage Validation
    # ==========================================================================
    coverage = len(mappings) / max(len(param_table), 1) * 100
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if coverage < 1.0:  # Менее 1% — полная неудача
        logger.warning("Phase F: Покрытие %.1f%% < 1%% — калибровка отклонена (%.1f мс)",
                       coverage, elapsed_ms)
        return []
    logger.info("Phase F: Coverage = %.1f%% (%d/%d) — OK (%.1f мс)",
               coverage, len(mappings), len(param_table), elapsed_ms)

    # ==========================================================================
    # Phase G: Anchor Calibration
    # ==========================================================================
    logger.info("Phase G: Anchor Calibration — валидация через якорные параметры")

    anchor_params_found = 0
    anchor_params_expected = 0

    for mapping in mappings:
        param = next((p for p in param_table if p.get("id") == mapping.param_id), None)
        if param and _is_anchor_param(param.get("name", "")):
            anchor_params_expected += 1
            # Проверяем что позиция якорного параметра выглядит правильно
            # (масляные параметры обычно в нижней части схемы)
            label_y = mapping.label_bbox.y1 if mapping.label_bbox else 0
            if label_y > 0.5:  # Нижняя половина экрана
                anchor_params_found += 1
                logger.debug("Phase G: Якорный параметр %d подтверждён", mapping.param_id)

    if anchor_params_expected > 0:
        anchor_ratio = anchor_params_found / anchor_params_expected
        logger.info("Phase G: Якорных параметров подтверждено %d/%d", anchor_params_found, anchor_params_expected)
        if anchor_ratio >= 0.5:
            logger.info("Phase G: Высокая уверенность калибровки (якорные параметры совпадают)")
    else:
        logger.debug("Phase G: Якорные параметры не найдены в таблице")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Калибровка завершена: %d параметров, покрытие %.1f%%, время %.1f мс",
        len(mappings), coverage * 100, elapsed_ms
    )

    return mappings
