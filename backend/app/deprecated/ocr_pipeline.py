"""OcrPipeline — двухпутевой OCR конвейер.

Оркестрирует все стадии:
1. PaddleOCR fast-path (detection + recognition за один вызов)
2. Florence-2 semantic path (OCR_WITH_REGION, по расписанию)
3. Layout Analysis (TextClassifier → ProximityGraph → PairExtractor)
4. Fusion (ResultMerger paddle + florence)
5. Confidence Scoring

Оптимизации производительности:
- Изменение размера кадра: до MAX_OCR_DIM (1280px по длинной стороне) перед OCR
- Обнаружение смены сцены: пропуск дублирующихся/статичных кадров
- Глобальный singleton OcrPipeline через get_ocr_pipeline()
- Florence: num_beams=3 (стандарт из документации Florence-2-large-ft)

Использование:
    pipeline = get_ocr_pipeline()
    result = pipeline.process(frame)
    print(result.raw_fields)  # {"TI-101": "758.3", ...}
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import cv2
import numpy as np

from app.core.confidence_scorer import ConfidenceScorer
from app.core.florence_detector import FlorenceDetector
from app.core.ocr_models import BBox, RecognitionResult, SourceType, TextBox, TextPair
from app.core.pair_extractor import PairExtractor
from app.core.result_merger import ResultMerger
from app.core.text_classifier import TextClassifier

logger = logging.getLogger(__name__)

# Максимальный размер кадра для OCR (по длинной стороне)
# PaddleOCR работает быстрее на меньших кадрах, качество не страдает до ~1280px
MAX_OCR_DIM = 1280

# Порог для обнаружения смены сцены (SSIM)
# Кадры с SSIM > этого значения считаются дубликатами
SCENE_CHANGE_THRESHOLD = 0.97

# Порог MSE для детекции дубликатов кадров (должен совпадать с pipeline.py)
# 120 — баланс между пропуском статичных кадров и детекцией изменений цифр
DUPLICATE_MSE_THRESHOLD = 120.0


class OcrPipeline:
    """Главный класс — двухпутевой OCR конвейер.

    Состояние между вызовами: кэш Florence, последние пары,
    предыдущий кадр для scene detection.
    """

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}

        # PaddleOCR (ленивая загрузка через существующий ocr_engine)
        self._paddle = None
        self._paddle_ready = False

        # Florence-2 (ленивая загрузка)
        self._florence: FlorenceDetector | None = None
        self._florence_interval = config.get("florence_interval_sec", 5.0)
        self._last_florence_ts = 0.0
        self._florence_pairs_cache: list[TextPair] | None = None

        # Layout Analysis
        self._extractor = PairExtractor()

        # Fusion
        self._merger = ResultMerger()
        self._scorer = ConfidenceScorer()

        # Счётчик кадров
        self._frame_idx = 0

        # Конфиг
        # Florence отключен по умолчанию для ускорения обработки (~6 сек/кадр -> ~0.3-0.5 сек/кадр)
        # Использовать config={"use_florence": True} только для калибровки
        self._use_florence = config.get("use_florence", False)

        # Scene change detection
        self._prev_gray: np.ndarray | None = None
        self._scene_skip_count = 0

        # Размер кадра для OCR (кэш)
        self._ocr_scale = 1.0

        # Адаптивное управление Florence: если Paddle уверен — пропускаем Florence
        self._paddle_confidence_history: list[float] = []
        self._paddle_confidence_window = 5  # окно для усреднения
        self._paddle_confidence_threshold = 0.9  # порог уверенности

    def _get_paddle(self):
        """Ленивая загрузка PaddleOCR через существующий ocr_engine."""
        if not self._paddle_ready:
            from app.core.ocr_engine import get_ocr_engine
            self._paddle = get_ocr_engine()
            self._paddle_ready = True
        return self._paddle

    def _get_florence(self) -> FlorenceDetector | None:
        """Ленивая загрузка Florence-2."""
        if not self._use_florence:
            return None
        if self._florence is None:
            if FlorenceDetector.is_available():
                self._florence = FlorenceDetector()
            else:
                logger.warning("Florence-2 недоступна (нет torch/transformers)")
                self._use_florence = False
                return None
        return self._florence

    @staticmethod
    def _resize_for_ocr(frame: np.ndarray, max_dim: int = MAX_OCR_DIM) -> tuple[np.ndarray, float]:
        """Уменьшает кадр до max_dim по длинной стороне для ускорения OCR.

        Args:
            frame: Входной кадр BGR.
            max_dim: Максимальный размер по длинной стороне.

        Returns:
            (resized_frame, scale_factor) — масштабированный кадр и коэффициент.
        """
        h, w = frame.shape[:2]
        max_side = max(h, w)
        if max_side <= max_dim:
            return frame, 1.0
        scale = max_dim / max_side
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale

    @staticmethod
    def _is_duplicate_frame(
        gray: np.ndarray,
        prev_gray: np.ndarray | None,
        threshold: float = SCENE_CHANGE_THRESHOLD,
    ) -> bool:
        """Проверяет, является ли кадр дубликатом предыдущего.

        Использует MSE (Mean Squared Error) на уменьшенных кадрах с ранним выходом.
        Для SCADA-экранов: если экран не изменился — OCR повторять не нужно.

        Оптимизация: вычисляет MSE построчно с ранним выходом — если накопленная
        ошибка превышает порог, немедленно возвращает "не дубликат", не вычисляя
        MSE для всего кадра. Это экономит время на явно разных кадрах.

        Args:
            gray: Текущий кадр в grayscale.
            prev_gray: Предыдущий кадр в grayscale.
            threshold: Не используется (оставлен для совместимости).

        Returns:
            True если кадр — дубликат (не нужно обрабатывать).
        """
        if prev_gray is None:
            return False
        if gray.shape != prev_gray.shape:
            return False

        # Быстрая проверка: MSE на уменьшенных кадрах с ранним выходом
        # Для SCADA-экранов это работает хорошо — статический контент
        small_h, small_w = 64, 64
        small_curr = cv2.resize(gray, (small_w, small_h))
        small_prev = cv2.resize(prev_gray, (small_w, small_h))

        # Ранний выход: вычисляем MSE построчно, останавливаемся если превысили порог
        # Порог нормализован на количество пикселей для раннего сравнения
        # total_pixels = small_h * small_w = 4096
        # max_accumulated_error = DUPLICATE_MSE_THRESHOLD * total_pixels
        max_accumulated_error = DUPLICATE_MSE_THRESHOLD * small_h * small_w

        accumulated_error = 0.0
        diff = small_curr.astype(np.float32) - small_prev.astype(np.float32)

        for row in diff:
            accumulated_error += np.sum(row * row)
            if accumulated_error >= max_accumulated_error:
                # Ранний выход: кадр точно не дубликат
                mse = accumulated_error / (small_h * small_w)
                logger.debug(
                    "OCR Pipeline: ранний выход, кадр уникальный (MSE=%.1f >= %.1f)",
                    mse, DUPLICATE_MSE_THRESHOLD
                )
                return False

        # Полный MSE для логирования
        mse = accumulated_error / (small_h * small_w)
        # MSE=0 → identical, MSE>150 → значение изменилось (для SCADA достаточно)
        # Для 8-bit: max MSE = 255^2 = 65025
        # Порог 150 — баланс между пропуском статичных кадров и детекцией изменений:
        # - Пропускает ~85% идентичных кадров (экономия времени)
        # - Детектирует изменения значений параметров (изменение цифр видно на MSE > 150)
        # Согласован с DUPLICATE_MSE_THRESHOLD из pipeline.py
        is_duplicate = mse < DUPLICATE_MSE_THRESHOLD
        if is_duplicate:
            logger.debug("OCR Pipeline: кадр классифицирован как дубликат (MSE=%.1f < %.1f)",
                        mse, DUPLICATE_MSE_THRESHOLD)
        else:
            logger.debug("OCR Pipeline: кадр классифицирован как уникальный (MSE=%.1f >= %.1f)",
                        mse, DUPLICATE_MSE_THRESHOLD)
        return is_duplicate

    @staticmethod
    def _compute_iou(box1: BBox, box2: BBox) -> float:
        """Вычисляет Intersection over Union (IoU) двух bounding box.

        Args:
            box1: Первый bounding box.
            box2: Второй bounding box.

        Returns:
            IoU в диапазоне [0.0, 1.0].
        """
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.w, box2.x + box2.w)
        y2 = min(box1.y + box1.h, box2.y + box2.h)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.w * box1.h
        area2 = box2.w * box2.h
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _deduplicate_boxes(
        self,
        paddle_boxes: list[TextBox],
        florence_boxes: list[TextBox],
        iou_threshold: float = 0.5,
    ) -> list[TextBox]:
        """Объединяет боксы Paddle и Florence, удаляя дубликаты по IoU.

        Если Florence бокс пересекается с Paddle (IoU > threshold),
        выбираем тот, у которого выше confidence.

        Args:
            paddle_boxes: Боксы от PaddleOCR.
            florence_boxes: Боксы от Florence-2.
            iou_threshold: Порог IoU для считания боксов дубликатами.

        Returns:
            Объединённый список боксов без дубликатов.
        """
        if not florence_boxes:
            return list(paddle_boxes)
        if not paddle_boxes:
            return list(florence_boxes)

        result = list(paddle_boxes)  # Начинаем с Paddle

        for fl_box in florence_boxes:
            is_duplicate = False
            for i, pd_box in enumerate(result):
                iou = self._compute_iou(fl_box.bbox, pd_box.bbox)
                if iou > iou_threshold:
                    # Дубликат: выбираем тот, у которого выше confidence
                    is_duplicate = True
                    if fl_box.confidence > pd_box.confidence:
                        result[i] = fl_box
                    break
            if not is_duplicate:
                result.append(fl_box)

        return result

    def process(
        self,
        frame: np.ndarray,
        stage_callback: Callable[[str], None] | None = None,
        preprocessed_scale: float | None = None,
    ) -> RecognitionResult:
        """Обрабатывает один кадр через полный конвейер.

        Оптимизации:
        - Изменение размера кадра до MAX_OCR_DIM для ускорения
        - Пропуск дублирующихся кадров (scene change detection)
        - Объединение боксов Paddle+Florence перед PairExtractor (один вызов)
        - При передаче preprocessed_scale пропускает повторное изменение размера

        Args:
            frame: Входной кадр BGR (может быть уже уменьшен препроцессором).
            stage_callback: Опциональный callback для отчёта о стадии обработки.
                Вызывается с именем стадии: 'ocr_paddle', 'ocr_florence',
                'ocr_layout', 'ocr_fusion', 'ocr_scoring'.
            preprocessed_scale: Если кадр уже был уменьшен препроцессором,
                передаём коэффициент масштабирования. В этом случае повторное
                уменьшение пропускается, а координаты масштабируются правильно.

        Returns:
            RecognitionResult с парами label:value и confidence.
        """
        t0 = time.perf_counter()

        # Scene change detection: пропускаем дубли
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._is_duplicate_frame(gray, self._prev_gray):
            self._scene_skip_count += 1
            self._prev_gray = gray
            # Возвращаем кэшированный результат с минимальной задержкой
            if hasattr(self, '_last_result') and self._last_result is not None:
                self._frame_idx += 1
                return RecognitionResult(
                    raw_fields=self._last_result.raw_fields,
                    confidence=self._last_result.confidence,
                    source=self._last_result.source,
                    pairs=self._last_result.pairs,
                    frame_idx=self._frame_idx,
                    processing_ms=0.1,  # мгновенно
                )
        self._prev_gray = gray

        # Уменьшение кадра для OCR (или использование уже уменьшенного)
        # ОПТИМИЗАЦИЯ: если кадр уже был уменьшен препроцессором, не делаем повторно
        if preprocessed_scale is not None and preprocessed_scale < 1.0:
            ocr_frame = frame
            scale = preprocessed_scale
            logger.debug("Используем предварительно уменьшенный кадр (scale=%.2f)", scale)
        else:
            ocr_frame, scale = self._resize_for_ocr(frame)
        self._ocr_scale = scale

        # Стадия 1: PaddleOCR fast-path (detection + recognition)
        if stage_callback:
            stage_callback("ocr_paddle")
        paddle_boxes = self._paddle_detect_recognize(ocr_frame)
        logger.debug("PaddleOCR: %d текстов распознано (scale=%.2f)", len(paddle_boxes), scale)

        # Масштабируем координаты обратно к полному размеру
        if scale != 1.0:
            for box in paddle_boxes:
                box.bbox = BBox(
                    x=int(box.bbox.x / scale),
                    y=int(box.bbox.y / scale),
                    w=int(box.bbox.w / scale),
                    h=int(box.bbox.h / scale),
                )

        # Собираем статистику уверенности Paddle для адаптивного управления Florence
        paddle_confidence = self._calculate_paddle_confidence_from_boxes(paddle_boxes)
        self._paddle_confidence_history.append(paddle_confidence)
        if len(self._paddle_confidence_history) > self._paddle_confidence_window:
            self._paddle_confidence_history.pop(0)

        # Стадия 2: Florence-2 semantic path (по расписанию ИЛИ если Paddle не уверен)
        florence_boxes: list[TextBox] = []
        florence = self._get_florence()
        now = time.time()

        # Адаптивная логика: если Paddle уверен (>0.9) последние 5 кадров — пропускаем Florence
        should_skip_florence = self._should_skip_florence()

        if florence is not None and not should_skip_florence and (now - self._last_florence_ts) >= self._florence_interval:
            if stage_callback:
                stage_callback("ocr_florence")
            fl_boxes = florence.propose(ocr_frame)
            if fl_boxes:
                # Масштабируем координаты Florence обратно
                if scale != 1.0:
                    for box in fl_boxes:
                        box.bbox = BBox(
                            x=int(box.bbox.x / scale),
                            y=int(box.bbox.y / scale),
                            w=int(box.bbox.w / scale),
                            h=int(box.bbox.h / scale),
                        )
                florence_boxes = fl_boxes
            self._last_florence_ts = now

        # Стадия 3: Fusion — объединяем боксы Paddle + Florence, удаляем дубликаты
        if stage_callback:
            stage_callback("ocr_fusion")
        combined_boxes = self._deduplicate_boxes(paddle_boxes, florence_boxes, iou_threshold=0.5)
        logger.debug("Combined boxes: %d (Paddle: %d, Florence: %d)",
                     len(combined_boxes), len(paddle_boxes), len(florence_boxes))

        # Стадия 4: Layout Analysis — TextClassifier → ProximityGraph → PairExtractor
        if stage_callback:
            stage_callback("ocr_layout")
        combined_pairs = self._extractor.extract(combined_boxes)
        logger.debug("PairExtractor: %d пар из объединённых боксов", len(combined_pairs))

        # Определяем source на основе вклада Florence
        source: SourceType = "merged" if florence_boxes else "paddle"

        # Стадия 5: Confidence scoring
        if stage_callback:
            stage_callback("ocr_scoring")
        confidence = self._calculate_paddle_confidence(combined_pairs)

        # Конвертируем пары в словарь
        raw_fields = self._pairs_to_dict(combined_pairs)

        processing_ms = (time.perf_counter() - t0) * 1000
        self._frame_idx += 1

        result = RecognitionResult(
            raw_fields=raw_fields,
            confidence=confidence,
            source=source,
            pairs=combined_pairs,
            frame_idx=self._frame_idx,
            processing_ms=round(processing_ms, 1),
        )

        # Кэшируем для дублирующихся кадров
        self._last_result = result

        return result

    def process_paddle_only(self, frame: np.ndarray) -> RecognitionResult:
        """Быстрый путь: только Paddle без Florence.

        Используется когда нужна максимальная скорость.
        """
        t0 = time.perf_counter()

        # Уменьшение кадра
        ocr_frame, scale = self._resize_for_ocr(frame)

        paddle_boxes = self._paddle_detect_recognize(ocr_frame)

        # Масштабируем координаты
        if scale != 1.0:
            for box in paddle_boxes:
                box.bbox = BBox(
                    x=int(box.bbox.x / scale),
                    y=int(box.bbox.y / scale),
                    w=int(box.bbox.w / scale),
                    h=int(box.bbox.h / scale),
                )

        paddle_pairs = self._extractor.extract(paddle_boxes)
        raw_fields = ResultMerger._pairs_to_dict(paddle_pairs)
        confidence = self._calculate_paddle_confidence(paddle_pairs)

        processing_ms = (time.perf_counter() - t0) * 1000
        self._frame_idx += 1

        return RecognitionResult(
            raw_fields=raw_fields,
            confidence=confidence,
            source="paddle",
            pairs=paddle_pairs,
            frame_idx=self._frame_idx,
            processing_ms=round(processing_ms, 1),
        )

    def invalidate_florence_cache(self) -> None:
        """Сбрасывает кэш Florence (например, при смене мнемосхемы)."""
        self._florence_pairs_cache = None
        self._last_florence_ts = 0.0
        self._paddle_confidence_history.clear()

    def _calculate_paddle_confidence(self, pairs: list[TextPair]) -> float:
        """Вычисляет среднюю уверенность PaddleOCR по всем парам.

        Args:
            pairs: Список распознанных пар label-value.

        Returns:
            Средняя уверенность в диапазоне [0.0, 1.0].
        """
        if not pairs:
            return 0.0

        total_confidence = 0.0
        count = 0
        for pair in pairs:
            if pair.label:
                total_confidence += pair.label.confidence
                count += 1
            if pair.value:
                total_confidence += pair.value.confidence
                count += 1

        return total_confidence / count if count > 0 else 0.0

    def _calculate_paddle_confidence_from_boxes(self, boxes: list[TextBox]) -> float:
        """Вычисляет среднюю уверенность PaddleOCR по боксам.

        Args:
            boxes: Список текстовых боксов.

        Returns:
            Средняя уверенность в диапазоне [0.0, 1.0].
        """
        if not boxes:
            return 0.0

        total_confidence = sum(box.confidence for box in boxes)
        return total_confidence / len(boxes)

    @staticmethod
    def _pairs_to_dict(pairs: list[TextPair]) -> dict[str, str]:
        """Конвертирует список пар в словарь {label: value}."""
        result: dict[str, str] = {}
        for pair in pairs:
            label = pair.label.text.strip()
            value = pair.value.text.strip()
            if label and value:
                result[label] = value
        return result

    def _should_skip_florence(self) -> bool:
        """Определяет, нужно ли пропустить Florence на этом кадре.

        Пропускаем Florence если Paddle уверен (>0.9) последние N кадров.
        Это экономит время на статичных SCADA-экранах.

        Returns:
            True если Florence можно пропустить.
        """
        if len(self._paddle_confidence_history) < self._paddle_confidence_window:
            return False  # Недостаточно данных

        avg_confidence = sum(self._paddle_confidence_history) / len(self._paddle_confidence_history)
        return avg_confidence >= self._paddle_confidence_threshold

    @property
    def scene_skip_count(self) -> int:
        """Количество пропущенных дублирующихся кадров."""
        return self._scene_skip_count

    # -----------------------------------------------------------------------
    # PaddleOCR integration
    # -----------------------------------------------------------------------

    def _paddle_detect_recognize(self, frame: np.ndarray) -> list[TextBox]:
        """Запускает PaddleOCR и конвертирует результат в TextBox[].

        Использует существующий ocr_engine.ocr_full_frame() для совместимости
        с PaddleOCR 2.9+ (OCRResult format) и legacy format.
        """
        from app.core.ocr_engine import _parse_ocr_results, get_ocr_engine

        ocr = self._get_paddle()
        if ocr is None:
            return []

        try:
            results = ocr.predict(frame)
        except Exception as e:
            logger.error("PaddleOCR ошибка: %s", e)
            return []

        h, w = frame.shape[:2]
        
        try:
            ocr_results = _parse_ocr_results(results, h, w)
        except Exception as parse_err:
            logger.warning("_parse_ocr_results ошибка (frame %dx%d): %s", w, h, parse_err)
            return []

        # Конвертируем OCRTextResult → TextBox
        text_boxes: list[TextBox] = []
        for r in ocr_results:
            try:
                # Нормализованные → пиксельные координаты
                x1 = int(r.bbox.x1 * w)
                y1 = int(r.bbox.y1 * h)
                x2 = int(r.bbox.x2 * w)
                y2 = int(r.bbox.y2 * h)
                bbox = BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

                text_boxes.append(
                    TextBox(
                        bbox=bbox,
                        text=r.text,
                        confidence=r.confidence,
                        source="paddle",
                    )
                )
            except (AttributeError, TypeError) as box_err:
                logger.debug("Пропуск невалидного OCR результата: %s", box_err)
                continue

        return text_boxes
