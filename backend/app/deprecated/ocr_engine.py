"""OCR-движок на базе PaddleOCR PP-OCRv5 + Florence-2.

Три режима распознавания:
1. Full-frame — для калибровки (находит все тексты)
2. ROI-based — для производства (только значения в заданных областях)
3. Enhanced — двухпутевой пайплайн Paddle + Florence + Layout Analysis

Обязательно lang='ru' для распознавания кириллицы.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Отключаем OneDNN (MKL-DNN) — в PaddlePaddle 3.x на Windows
# OneDNN вызывает layout-конфликты при CPU-inference.
# Должно быть установлено ДО импорта paddle.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["MKLDNN_DISABLE"] = "1"

# PaddleOCR 2.9+ через PaddleX тянет modelscope -> torch.
# Если paddle загрузится первым, DLL torch испортится.
# Поэтому torch должен быть импортирован ДО paddle.
try:
    import torch  # noqa: F401 — preload до paddle
except ImportError:
    pass

import cv2
import numpy as np

# Принудительно отключаем OneDNN через paddle API после импорта
try:
    import paddle
    paddle.set_flags({"FLAGS_use_mkldnn": False})

    # PaddleX внутри явно вызывает config.enable_mkldnn() при создании
    # inference-предикторов. Это игнорирует глобальный FLAGS_use_mkldnn.
    # Патчим enable_mkldnn → no-op, чтобы предикторы НЕ использовали OneDNN.
    try:
        _cfg_cls = paddle.inference.Config
        _cfg_cls.enable_mkldnn = lambda self, *a, **kw: None
        if hasattr(_cfg_cls, "enable_mkldnn_bfloat16"):
            _cfg_cls.enable_mkldnn_bfloat16 = lambda self, *a, **kw: None
        logger_pre = logging.getLogger(__name__)
        logger_pre.info("paddle.inference.Config.enable_mkldnn patched → no-op")
    except Exception as _patch_err:
        logger_pre = logging.getLogger(__name__)
        logger_pre.warning("Не удалось патчить enable_mkldnn: %s", _patch_err)
except Exception:
    pass

from app.config import settings
from app.models.schemas import BoundingBox, OCRTextResult, ZoneType

logger = logging.getLogger(__name__)

# Глобальный экземпляр OCR (lazy init)
_ocr_instance = None

# Whitelist известных параметров PaddleOCR 3.x
# PaddleOCR 3.x имеет явную сигнатуру с именованными параметрами.
# Старые параметры 2.x (det_db_thresh, rec_batch_num, use_dilation) УДАЛЕНЫ.
_KNOWN_PADDLE_PARAMS: frozenset[str] = frozenset({
    # Параметры детекции текста (PaddleOCR 3.x)
    "text_det_thresh", "text_det_box_thresh", "text_det_unclip_ratio",
    "text_det_limit_side_len", "text_det_limit_type", "text_det_input_shape",
    # Параметры распознавания (PaddleOCR 3.x)
    "text_recognition_batch_size", "text_rec_score_thresh", "text_rec_input_shape",
    # Ориентация текстовых строк
    "use_textline_orientation", "textline_orientation_batch_size",
    # Документы (ориентация, разгибание)
    "use_doc_orientation_classify", "use_doc_unwarping",
    # Пути к моделям (PaddleOCR 3.x)
    "text_detection_model_dir", "text_detection_model_name",
    "text_recognition_model_dir", "text_recognition_model_name",
    "textline_orientation_model_dir", "textline_orientation_model_name",
    # Общие параметры
    "lang", "ocr_version", "return_word_box",
})


def _try_add_ocr_param(init_kwargs: dict, param_name: str, value) -> bool:
    """Добавляет параметр в init_kwargs, проверяя по whitelist.

    PaddleOCR.__init__ использует **kwargs для передачи параметров
    в детектор и распознаватель, поэтому inspect.signature() не работает.
    Вместо этого проверяем по известному списку параметров.

    Args:
        init_kwargs: Словарь параметров инициализации PaddleOCR.
        param_name: Имя параметра для добавления.
        value: Значение параметра.

    Returns:
        True если параметр добавлен успешно, False если не поддерживается.
    """
    if param_name in _KNOWN_PADDLE_PARAMS:
        init_kwargs[param_name] = value
        logger.debug("Параметр OCR '%s'=%s добавлен", param_name, value)
        return True
    else:
        logger.warning(
            "Параметр OCR '%s' не найден в whitelist известных параметров",
            param_name
        )
        return False


def get_ocr_engine():
    """Возвращает singleton-экземпляр PaddleOCR.

    Инициализирует PaddleOCR 3.x с оптимизированными параметрами для SCADA:
    - text_det_thresh=0.2: снижен с 0.3 для обнаружения мелкого текста
    - text_det_unclip_ratio=2.0: расширен с 1.5 для полных символов
    - text_recognition_batch_size=6: батч-распознавание для нескольких регионов
    - use_textline_orientation=True: ориентация текстовых строк

    Returns:
        Экземпляр PaddleOCR.
    """
    global _ocr_instance
    if _ocr_instance is None:
        # Подавляем шумные логи faiss при импорте paddleocr
        import logging
        logging.getLogger("faiss").setLevel(logging.ERROR)

        from paddleocr import PaddleOCR

        # PaddleOCR 2.9+: use_angle_cls переименован в use_textline_orientation
        # Но не все версии поддерживают этот параметр — пробуем с fallback
        init_kwargs: dict = {
            "lang": settings.ocr_lang,  # 'ru' — КРИТИЧЕСКИ ВАЖНО
        }

        # Параметры настройки для SCADA (PaddleOCR 3.x)
        # use_textline_orientation=True — определение ориентации текстовых строк
        _try_add_ocr_param(init_kwargs, "use_textline_orientation", True)
        # text_det_thresh — порог бинаризации карты детекции (default 0.3)
        _try_add_ocr_param(init_kwargs, "text_det_thresh", settings.ocr_det_db_thresh)
        # text_det_unclip_ratio — расширение bbox для полных символов (default 1.5)
        _try_add_ocr_param(init_kwargs, "text_det_unclip_ratio", settings.ocr_det_db_unclip_ratio)
        # text_recognition_batch_size — размер батча распознавания (default 1)
        _try_add_ocr_param(init_kwargs, "text_recognition_batch_size", settings.ocr_rec_batch_num)
        # ПРИМЕЧАНИЕ: use_dillation удалён в PaddleOCR 3.x

        # Модели из локальной директории (если есть)
        # PaddleOCR 3.x использует text_detection_model_dir вместо det_model_dir
        det_dir = Path(settings.models_dir) / "det"
        rec_dir = Path(settings.models_dir) / "rec"
        cls_dir = Path(settings.models_dir) / "cls"
        if det_dir.exists():
            init_kwargs["text_detection_model_dir"] = str(det_dir)
        if rec_dir.exists():
            init_kwargs["text_recognition_model_dir"] = str(rec_dir)
        if cls_dir.exists():
            init_kwargs["textline_orientation_model_dir"] = str(cls_dir)

        # Пробуем инициализировать PaddleOCR, последовательно убирая
        # неподдерживаемые параметры при ошибках
        # PaddleOCR 3.x выбрасывает ValueError для неизвестных параметров
        _optional_params = [
            "use_textline_orientation", "text_det_thresh",
            "text_det_unclip_ratio", "text_recognition_batch_size",
            "text_detection_model_dir", "text_recognition_model_dir",
            "textline_orientation_model_dir",
        ]
        while True:
            try:
                _ocr_instance = PaddleOCR(**init_kwargs)
                break
            except ValueError as e:
                # PaddleOCR 3.x: ValueError("Unknown argument: <param>")
                err_msg = str(e)
                removed = False
                for param in _optional_params:
                    if param in err_msg and param in init_kwargs:
                        logger.warning(
                            "PaddleOCR не поддерживает '%s', убираем: %s",
                            param, e
                        )
                        init_kwargs.pop(param)
                        removed = True
                        break
                if not removed:
                    # Неизвестная ошибка — пробуем с минимальными параметрами
                    logger.error(
                        "PaddleOCR: неизвестная ValueError, пробуем минимальную конфигурацию: %s", e
                    )
                    init_kwargs = {"lang": settings.ocr_lang}
                    _ocr_instance = PaddleOCR(**init_kwargs)
                    break
            except TypeError as e:
                # Fallback для других типов ошибок
                logger.error(
                    "PaddleOCR: TypeError при инициализации: %s", e
                )
                init_kwargs = {"lang": settings.ocr_lang}
                _ocr_instance = PaddleOCR(**init_kwargs)
                break
        logger.info("PaddleOCR инициализирован (lang=%s, params=%s)",
                   settings.ocr_lang,
                   {k: v for k, v in init_kwargs.items() if k != "lang"})
    return _ocr_instance


def warmup_ocr_engine() -> float:
    """Прогревает OCR-движок для JIT-компиляции CUDA-ядер.

    Создаёт тестовые кадры и запускает OCR для инициализации CUDA-ядер,
    кэшей моделей и предварительного выделения GPU-памяти.
    Вызывается при старте приложения для предотвращения
    задержек на первом реальном запросе.

    Returns:
        Время прогрева в секундах.
    """
    import time

    ocr = get_ocr_engine()
    if ocr is None:
        logger.warning("warmup_ocr_engine: OCR не инициализирован")
        return 0.0

    logger.info("Прогрев PaddleOCR (JIT-компиляция CUDA-ядер + GPU memory allocation)...")
    t0 = time.perf_counter()

    # Создаём тестовые кадры разных размеров для прогрева
    # Маленький кадр для быстрой инициализации
    dummy_small = np.zeros((100, 100, 3), dtype=np.uint8)
    # Средний кадр для предварительного выделения памяти (типичный размер ROI)
    dummy_medium = np.zeros((200, 400, 3), dtype=np.uint8)

    try:
        # Прогрев 1: маленький кадр для JIT-компиляции
        ocr.predict(dummy_small)

        # Прогрев 2: средний кадр для предварительного выделения памяти
        ocr.predict(dummy_medium)

        # Прогрев 3: синхронизация CUDA для точного замера
        try:
            import paddle
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.synchronize()
        except Exception:
            pass  # Синхронизация не критична

        elapsed = time.perf_counter() - t0
        logger.info("PaddleOCR прогрет за %.2fs (GPU memory allocated)", elapsed)
        return elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.warning("Ошибка при прогреве PaddleOCR (%.2fs): %s", elapsed, e)
        return elapsed


def _parse_ocr_results(results, frame_h: int, frame_w: int) -> list[OCRTextResult]:
    """Парсит результаты PaddleOCR в единый формат.

    PaddleOCR 3.x (PaddleX) возвращает список OCRResult с ключами:
      rec_texts, rec_scores, rec_polys (или dt_polys).
    Legacy версии возвращали list[list[bbox, (text, conf)]]].

    Args:
        results: Сырой результат ocr.predict().
        frame_h: Высота кадра.
        frame_w: Ширина кадра.

    Returns:
        Список OCRTextResult.
    """
    if not results:
        return []

    first = results[0]
    if first is None:
        return []

    text_results: list[OCRTextResult] = []

    # PaddleOCR 3.x (PaddleX): OCRResult — dict-like объект
    if hasattr(first, "get") and ("rec_texts" in first or "rec_polys" in first):
        rec_texts = first.get("rec_texts", []) if hasattr(first, "get") else []
        rec_scores = first.get("rec_scores", []) if hasattr(first, "get") else []
        # Используем rec_polys (после ориентации) или dt_polys (сырые полигоны)
        polys = first.get("rec_polys", []) if hasattr(first, "get") else []
        if not polys:
            polys = first.get("dt_polys", []) if hasattr(first, "get") else []

        for i, text in enumerate(rec_texts):
            confidence = rec_scores[i] if i < len(rec_scores) else 0.0
            poly = polys[i] if i < len(polys) else None

            if poly is not None and len(poly) >= 2:
                # poly shape: (N, 2) — массив точек полигона
                xs = poly[:, 0].astype(float)
                ys = poly[:, 1].astype(float)
                x1 = xs.min() / frame_w
                y1 = ys.min() / frame_h
                x2 = xs.max() / frame_w
                y2 = ys.max() / frame_h
            else:
                x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0

            text_results.append(
                OCRTextResult(
                    text=text,
                    confidence=float(confidence),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return text_results

    # Legacy формат: list[list[bbox, (text, conf)]]]
    # Также может быть: list[bbox_points] с вложенной структурой
    for line in first:
        try:
            # Пропускаем если line — не список/кортеж ожидаемой структуры
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            bbox_pts = line[0]
            # line[1] должен быть [text, confidence] — проверяем структуру
            text_conf = line[1]
            if not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
                continue
            text = text_conf[0]
            confidence = text_conf[1]
            # Валидация: text должен быть строкой, confidence — числом
            if not isinstance(text, str) or not isinstance(confidence, (int, float)):
                continue
        except (IndexError, TypeError):
            continue

        x1 = min(p[0] for p in bbox_pts) / frame_w
        y1 = min(p[1] for p in bbox_pts) / frame_h
        x2 = max(p[0] for p in bbox_pts) / frame_w
        y2 = max(p[1] for p in bbox_pts) / frame_h

        text_results.append(
            OCRTextResult(
                text=text,
                confidence=float(confidence),
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            )
        )

    return text_results


def _filter_by_confidence(
    results: list[OCRTextResult],
    min_confidence: float | None = None,
) -> list[OCRTextResult]:
    """Фильтрует результаты OCR по уверенности.

    Удаляет результаты с уверенностью ниже порога и логирует
    низкоуверенные результаты для отладки.

    Args:
        results: Список результатов OCR.
        min_confidence: Минимальный порог уверенности (default из settings).

    Returns:
        Отфильтрованный список результатов.
    """
    if min_confidence is None:
        min_confidence = settings.ocr_low_confidence_threshold

    high_threshold = settings.ocr_high_confidence_threshold
    filtered: list[OCRTextResult] = []
    low_conf_count = 0

    for r in results:
        if r.confidence < min_confidence:
            logger.debug("Отфильтрован низкоуверенный результат: '%s' (conf=%.3f)",
                        r.text[:20] if len(r.text) > 20 else r.text, r.confidence)
            continue

        if r.confidence < high_threshold:
            low_conf_count += 1
            logger.debug("Низкая уверенность: '%s' (conf=%.3f)",
                        r.text[:30] if len(r.text) > 30 else r.text, r.confidence)

        filtered.append(r)

    if low_conf_count > 0:
        logger.info("Результатов с низкой уверенностью (%.2f-%.2f): %d из %d",
                   min_confidence, high_threshold, low_conf_count, len(results))

    return filtered


def ocr_full_frame(frame: np.ndarray) -> list[OCRTextResult]:
    """Распознаёт весь текст на кадре (режим калибровки).

    Выполняет полный OCR с фильтрацией по уверенности.
    Результаты с confidence < 0.6 отбрасываются.
    Результаты с confidence 0.6-0.8 логируются как низкоуверенные.

    PaddleOCR 3.x: использует predict() вместо устаревшего ocr().

    Args:
        frame: Входной кадр BGR.

    Returns:
        Список распознанных текстов с координатами и уверенностью.
    """
    ocr = get_ocr_engine()
    # PaddleOCR 3.x: ocr() устарел, используем predict()
    results = ocr.predict(frame)

    h, w = frame.shape[:2]
    parsed = _parse_ocr_results(results, h, w)

    # Фильтрация по уверенности
    filtered = _filter_by_confidence(parsed)

    if len(filtered) < len(parsed):
        logger.debug("OCR full-frame: %d -> %d результатов после фильтрации",
                    len(parsed), len(filtered))

    return filtered


def ocr_roi_single(frame_crop: np.ndarray, min_confidence: float = 0.5) -> tuple[str, float]:
    """Распознаёт текст в малой области (один ROI).

    Использует пониженный порог уверенности (0.5 вместо 0.6),
    так как ROI-кропы должны содержать чистый текст.

    Args:
        frame_crop: Вырезанная область кадра BGR.
        min_confidence: Минимальный порог уверенности (default 0.5).

    Returns:
        Кортеж (best_text, confidence) — результат с наивысшей уверенностью.
        Если нет результатов выше порога — возвращает ("", 0.0).
    """
    if frame_crop.size == 0:
        return ("", 0.0)

    ocr = get_ocr_engine()
    results = ocr.predict(frame_crop)

    ch, cw = frame_crop.shape[:2]
    parsed = _parse_ocr_results(results, ch, cw)

    if not parsed:
        return ("", 0.0)

    # Фильтруем по уверенности
    filtered = [r for r in parsed if r.confidence >= min_confidence]

    if not filtered:
        return ("", 0.0)

    # Берём результат с наивысшей уверенностью
    best = max(filtered, key=lambda r: r.confidence)
    return (best.text, best.confidence)


def ocr_roi_batch(
    frame: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    min_confidence: float = 0.5,
) -> list[tuple[str, float]]:
    """Распознаёт текст в нескольких областях (batch ROI-режим).

    Для каждого ROI (x, y, w, h) вырезает область из кадра
    и вызывает ocr_roi_single.

    Args:
        frame: Входной кадр BGR.
        rois: Список областей в формате (x, y, w, h) в пикселях.
        min_confidence: Минимальный порог уверенности (default 0.5).

    Returns:
        Список кортежей (text, confidence) для каждого ROI.
    """
    results: list[tuple[str, float]] = []

    for x, y, w, h in rois:
        if w <= 0 or h <= 0:
            results.append(("", 0.0))
            continue

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)

        if x2 - x1 < 5 or y2 - y1 < 5:
            results.append(("", 0.0))
            continue

        crop = frame[y1:y2, x1:x2]
        text, conf = ocr_roi_single(crop, min_confidence)
        results.append((text, conf))

    return results


def validate_parameter_value(text: str, short_name: str = "", unit: str = "") -> tuple[str, bool]:
    """Валидирует значение параметра по формату типа.

    Очищает текст и проверяет соответствие regex-паттерну
    для конкретного типа параметра SCADA.

    Args:
        text: Распознанный текст значения.
        short_name: Короткое имя типа (T, P, dP, Vb, L, n, Pos, V, f).
        unit: Единица измерения (для логирования).

    Returns:
        Кортеж (cleaned_text, is_valid).
    """
    import re

    if not text:
        return ("", False)

    # Очистка текста
    cleaned = text.strip()
    cleaned = cleaned.replace(",", ".")  # Замена запятой на точку
    cleaned = cleaned.rstrip(".")  # Удаление точки в конце

    # Regex-паттерны по типам параметров
    patterns: dict[str, str] = {
        "T": r"^-?\d{1,4}\.?\d{0,1}$",  # Температура: "234.5", "-12.3", "800"
        "P": r"^\d{1,5}\.?\d{0,2}$",  # Давление: "1234", "1234.56", "0"
        "dP": r"^-?\d{1,4}\.?\d{0,1}$",  # Перепад давления: "-12.5", "45.2"
        "Vb": r"^\d{1,3}\.\d{1,2}$",  # Вибрация: "12.34", "0.50"
        "L": r"^\d{1,4}\.?\d{0,1}$",  # Уровень: "500", "75.5"
        "n": r"^\d{1,5}$",  # Частота/обороты: "3000", "150"
        "Pos": r"^\d{1,3}\.?\d{0,1}$",  # Положение: "50.0", "100"
        "V": r"^\d{1,3}\.?\d{0,1}$",  # Напряжение: "220.5"
        "f": r"^\d{1,3}\.?\d{0,1}$",  # Влажность: "65.5"
    }

    # Выбор паттерна
    if short_name in patterns:
        pattern = patterns[short_name]
    else:
        # Дефолтный паттерн для неизвестных типов
        pattern = r"^-?\d{1,6}\.?\d{0,3}$"

    is_valid = bool(re.match(pattern, cleaned))

    if not is_valid:
        logger.debug(
            "Валидация значения не пройдена: '%s' -> '%s' (тип=%s, pattern=%s)",
            text, cleaned, short_name if short_name else "unknown", pattern
        )

    return (cleaned, is_valid)


def ocr_roi(
    frame: np.ndarray,
    roi: BoundingBox,
    rec_only_threshold: tuple[int, int] = (400, 100),
) -> OCRTextResult | None:
    """Распознаёт текст в заданной области (ROI-режим производства).

    Оптимизация для малых ROI (value boxes):
    - Если crop < 400px ширина и < 100px высота, используется rec-only режим
      (пропуск детекции, только распознавание) для ускорения.
    - В rec-only режиме добавляется 20% padding для лучшего распознавания.
    - Для PaddleOCR 2.9+ с det=False или методом rec() используется прямой вызов.

    Фильтрация по уверенности:
    - Результаты с confidence < 0.6 отбрасываются.
    - Низкоуверенные результаты (0.6-0.8) логируются.

    Args:
        frame: Входной кадр BGR.
        roi: Область интереса в нормализованных координатах.
        rec_only_threshold: Пороги (width, height) для rec-only режима.

    Returns:
        Распознанный текст или None.
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(roi.x1 * w))
    y1 = max(0, int(roi.y1 * h))
    x2 = min(w, int(roi.x2 * w))
    y2 = min(h, int(roi.y2 * h))

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < 5 or crop_h < 5:
        return None

    # Определяем режим: rec-only для малых ROI
    use_rec_only = crop_w < rec_only_threshold[0] and crop_h < rec_only_threshold[1]

    # Добавляем 20% padding для rec-only режима
    if use_rec_only:
        pad_w = int(crop_w * 0.2)
        pad_h = int(crop_h * 0.2)
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)
        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        logger.debug("OCR ROI rec-only: crop=%dx%d, padded=%dx%d",
                    crop_w, crop_h, x2_pad - x1_pad, y2_pad - y1_pad)
    else:
        crop = frame[y1:y2, x1:x2]
        logger.debug("OCR ROI full: crop=%dx%d", crop_w, crop_h)

    ocr = get_ocr_engine()

    # Пытаемся использовать rec-only режим если доступен
    # PaddleOCR 3.x: predict() поддерживает параметры детекции на лету
    if use_rec_only:
        try:
            # Для rec-only устанавливаем очень высокий text_det_thresh
            # чтобы детектор не нашёл "текст" в маленьком crop
            results = ocr.predict(crop, text_det_thresh=0.9)
        except (TypeError, ValueError):
            # Fallback: используем predict без параметров
            logger.debug("text_det_thresh не поддерживается, fallback к стандартному OCR")
            results = ocr.predict(crop)
    else:
        results = ocr.predict(crop)

    ch, cw = crop.shape[:2]
    parsed = _parse_ocr_results(results, ch, cw)

    if not parsed:
        return None

    # Фильтрация по уверенности
    filtered = _filter_by_confidence(parsed)

    if not filtered:
        return None

    # Берём результат с наивысшей уверенностью
    best = max(filtered, key=lambda r: r.confidence)

    return OCRTextResult(
        text=best.text,
        confidence=best.confidence,
        bbox=roi,
    )


# ---------------------------------------------------------------------------
# Глобальный экземпляр OcrPipeline (lazy init)
# ---------------------------------------------------------------------------
_ocr_pipeline_instance = None


def get_ocr_pipeline():
    """Возвращает singleton-экземпляр OcrPipeline (Paddle + Florence + Layout).

    Returns:
        Экземпляр OcrPipeline.
    """
    global _ocr_pipeline_instance
    if _ocr_pipeline_instance is None:
        from app.core.ocr_pipeline import OcrPipeline
        # Интервал 10 секунд — Florence дорогая, значения SCADA меняются медленно
        _ocr_pipeline_instance = OcrPipeline({"use_florence": True, "florence_interval_sec": 5.0})
        logger.info("OcrPipeline инициализирован (Paddle + Florence + Layout, interval=5s)")
    return _ocr_pipeline_instance


def ocr_full_frame_enhanced(frame: np.ndarray, stage_callback: Callable[[str], None] | None = None) -> dict:
    """Распознаёт текст на кадре через двухпутевой пайплайн.

    Paddle (быстрый) + Florence (семантический) + Layout Analysis.
    Возвращает RecognitionResult с парами label:value и confidence.

    Args:
        frame: Входной кадр BGR.
        stage_callback: Опциональный callback для отчёта о стадии обработки.
            Вызывается с: 'ocr_paddle', 'ocr_florence', 'ocr_fusion',
            'ocr_layout', 'ocr_scoring'.

    Returns:
        RecognitionResult с raw_fields, pairs, confidence.
    """
    pipeline = get_ocr_pipeline()
    return pipeline.process(frame, stage_callback=stage_callback)


def detect_color_state(
    frame: np.ndarray,
    bbox: BoundingBox,
) -> str:
    """Определяет цветовое состояние значения (normal/alarm/warning/inactive).

    Анализирует доминирующий цвет в области значения.

    Args:
        frame: Кадр BGR.
        bbox: Область значения в нормализованных координатах.

    Returns:
        Строка состояния: 'normal' | 'alarm' | 'warning' | 'inactive'.
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox.x1 * w))
    y1 = max(0, int(bbox.y1 * h))
    x2 = min(w, int(bbox.x2 * w))
    y2 = min(h, int(bbox.y2 * h))

    if x2 - x1 < 2 or y2 - y1 < 2:
        return "normal"

    region = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Зелёный: H 35-85, S > 50, V > 50
    green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    green_ratio = green_mask.sum() / 255 / (region.shape[0] * region.shape[1])

    # Красный: H 0-10 или 170-180, S > 50, V > 50
    red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    red_ratio = (red_mask1.sum() + red_mask2.sum()) / 255 / (region.shape[0] * region.shape[1])

    # Жёлтый/оранжевый: H 10-35, S > 50, V > 50
    yellow_mask = cv2.inRange(hsv, (10, 50, 50), (35, 255, 255))
    yellow_ratio = yellow_mask.sum() / 255 / (region.shape[0] * region.shape[1])

    # Классификация
    if red_ratio > 0.15:
        return "alarm"
    elif yellow_ratio > 0.15:
        return "warning"
    elif green_ratio > 0.15:
        return "normal"
    else:
        return "inactive"


class OcrEngine:
    """Обертка-класс для OCR-движка.

    Предоставляет объектно-ориентированный интерфейс к функциям OCR.
    Делегирует вызовы к существующим функциям модуля.
    """

    def __init__(self) -> None:
        """Инициализирует OCR-движок (lazy init при первом использовании)."""
        self._engine = None

    def _get_engine(self):
        """Возвращает инициализированный экземпляр PaddleOCR."""
        if self._engine is None:
            self._engine = get_ocr_engine()
        return self._engine

    def recognize_full_frame(self, frame: np.ndarray) -> list[OCRTextResult]:
        """Распознаёт весь текст на кадре.

        Args:
            frame: Входной кадр BGR.

        Returns:
            Список распознанных текстов с координатами и уверенностью.
        """
        return ocr_full_frame(frame)

    def recognize_roi(self, frame: np.ndarray, roi: BoundingBox) -> OCRTextResult | None:
        """Распознаёт текст в заданной области (ROI).

        Args:
            frame: Входной кадр BGR.
            roi: Область интереса в нормализованных координатах.

        Returns:
            Распознанный текст или None.
        """
        return ocr_roi(frame, roi)

    def recognize_enhanced(self, frame: np.ndarray) -> dict:
        """Распознаёт текст на кадре через двухпутевой пайплайн.

        Args:
            frame: Входной кадр BGR.

        Returns:
            RecognitionResult с raw_fields, pairs, confidence.
        """
        return ocr_full_frame_enhanced(frame)

    def warmup(self) -> float:
        """Прогревает OCR-движок для JIT-компиляции CUDA-ядер.

        Returns:
            Время прогрева в секундах.
        """
        return warmup_ocr_engine()
