"""Главный конвейер обработки видео.

Оркестрирует все этапы:
1. Извлечение кадров
2. Предобработка
3. Сегментация зон
4. OCR
5. Кластеризация label-value
6. Калибровка / распознавание
7. Обработка значений
8. Генерация XML
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

# Порог MSE для детекции дубликатов кадров (fallback при отсутствии калибровки)
# Был 500 — слишком высокий, пропускались кадры с изменившимися значениями параметров
DUPLICATE_MSE_THRESHOLD = 120.0

# Порог MSE для детекции смены сцены/вкладки (верхняя часть экрана)
# Ниже, чем DUPLICATE_MSE_THRESHOLD, т.к. изменение вкладки — это серьёзное событие
SCENE_CHANGE_MSE_THRESHOLD = 50.0

# Доля верхней части кадра для проверки смены сцены (где находятся вкладки SCADA)
SCENE_CHANGE_TOP_REGION_RATIO = 0.1

# === Tier 1: Quick pixel difference threshold ===
# Минимальное количество изменившихся пикселей для раннего выхода
PIXEL_DIFF_MIN = 20
# Порог бинаризации для подсчёта изменившихся пикселей
PIXEL_DIFF_THRESHOLD = 25

# === Tier 2: ROI-based MSE thresholds ===
# Порог MSE для отдельного ROI — изменение цифры в ROI 50x30 даёт MSE ~15-30
ROI_MSE_THRESHOLD = 8.0

# === Adaptive Frame Skipping Configuration ===
# Начальный интервал пропуска кадров (1 = обрабатывать каждый кадр)
ADAPTIVE_SKIP_INITIAL = 1
# Максимальный интервал пропуска (8 = обрабатывать каждый 8-й кадр)
ADAPTIVE_SKIP_MAX = 4
# Множитель увеличения интервала (2x каждый раз)
ADAPTIVE_SKIP_MULTIPLIER = 2
# Количество последовательных дубликатов для увеличения интервала
ADAPTIVE_SKIP_DUPLICATE_THRESHOLD = 3

# === Forced Processing Configuration ===
# Минимальный интервал между обработкой кадров (в секундах)
MIN_PROCESS_INTERVAL_SEC = 2.0
# Количество кадров после смены вкладки, которые всегда обрабатываются
FORCE_PROCESS_AFTER_TAB_CHANGE = 3
# Максимальное количество последовательных дубликатов перед принудительной обработкой
MAX_CONSECUTIVE_DUPLICATES = 10

# === Parallel Pipeline Configuration ===
# Максимальный размер очередей между стадиями
PARALLEL_QUEUE_SIZE = 8
# Количество потоков для постобработки (CPU-bound)
PARALLEL_POSTPROCESS_WORKERS = 2
# Количество потоков для параллельной предобработки кадров (CPU-bound)
PREPROCESS_WORKERS = 8
# Sentinel для сигнала завершения
_SENTINEL = None

from app.config import settings
from app.core.calibration import CalibrationProfile, calibrate_with_grounding, match_labels_to_params, _compute_match_score
from app.core.color_filter import ColorFilter
from app.core.confidence_scorer import ConfidenceScorer
from app.core.frame_preprocessor import (
    increment_cache_epoch,
    preprocess_frame,
)
from app.core.ocr_engine import (
    detect_color_state,
    ocr_full_frame,
    ocr_roi,
    ocr_roi_batch,
    validate_parameter_value,
)
from app.core.ocr_logger import OcrLogger
from app.core.parameter_mapper import load_parameter_table
from app.core.screen_detector import (
    ScadaTabTracker,
    detect_active_scada_tab,
    has_tab_changed,
)
from app.core.spatial_clusterer import cluster_label_value_pairs, parse_right_panel
from app.core.value_processor import ValueProcessor
from app.core.video_ingestion import extract_frames
from app.core.xml_generator import create_snapshot, generate_xml
from app.core.zone_segmentor import (
    PopupDialogResult,
    detect_popup_dialog,
    extract_zone,
    invalidate_zone_cache,
    segment_zones,
)
from app.models.schemas import (
    BoundingBox,
    OCRTextResult,
    ParamMetadata,
    PipelineStatus,
    SnapshotData,
    VideoType,
    ZoneType,
)
from app.utils.xml_utils import format_timestamp

logger = logging.getLogger(__name__)


class Pipeline:
    """Конвейер обработки видео от кадра до XML."""

    def __init__(self) -> None:
        self.status = PipelineStatus(video_id="", status="idle")
        self._value_processor = ValueProcessor()
        self._calibration: CalibrationProfile | None = None
        self._param_table: list[dict] = []
        self._snapshots: list[SnapshotData] = []
        self._screen_corners: np.ndarray | None = None
        self._progress_callback: Callable[[int, int, str], None] | None = None
        # Для пропуска дублирующихся кадров
        self._prev_frame_gray: np.ndarray | None = None
        self._prev_params: dict[int, str] = {}
        self._duplicate_skip_count: int = 0
        # Florence verification (ленивая загрузка, отключена по умолчанию)
        self._florence_detector = None
        self._frame_counter: int = 0
        self._last_florence_text: str = ""
        # SCADA tab detection and tracking
        self._tab_tracker = ScadaTabTracker()
        self._current_sheet_name: str = ""
        self._tab_change_count: int = 0
        # Adaptive frame skipping state
        self._adaptive_skip_interval: int = ADAPTIVE_SKIP_INITIAL
        self._consecutive_duplicates: int = 0
        self._frames_since_last_process: int = 0
        # Frame skip statistics
        self._skip_stats: dict = {
            "total_frames": 0,
            "processed_frames": 0,
            "skipped_by_adaptive": 0,
            "skipped_by_duplicate": 0,
            "interval_changes": 0,
            "interval_history": [],
        }
        # Новые модули для улучшенной обработки
        self._color_filter = ColorFilter()
        self._scada_tab_tracker = ScadaTabTracker()
        self._confidence_scorer = ConfidenceScorer()
        # Порог confidence для фильтрации пар
        self._confidence_threshold: float = 0.3
        # Хранилище калибровок по вкладкам (per-tab calibration profiles)
        self._tab_calibrations: dict[str, CalibrationProfile] = {}
        # Счётчики для принудительной обработки кадров
        self._frames_since_tab_change: int = 999
        self._consecutive_duplicate_count: int = 0
        self._last_processed_timestamp: float = 0.0
        # Florence-based scene tracking
        self._current_tab: str | None = None
        self._current_gpa_type: str | None = None

    def process_video(
        self,
        video_path: str | Path,
        param_table_path: str | Path | None = None,
        video_type: VideoType | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        use_parallel: bool = True,
    ) -> tuple[str, str]:
        """Обрабатывает видеофайл и генерирует XML.

        Args:
            video_path: Путь к видеофайлу.
            param_table_path: Путь к таблице параметров (.xlsx/.csv). Если None,
                используется OCR-only режим без калибровки по ROI.
            video_type: Тип видео (если None, определяется автоматически).
            progress_callback: Функция обратного вызова для отчёта о прогрессе.
                Принимает (current_frame, total_frames, stage).
            use_parallel: Если True, использует параллельный конвейер обработки.
                Параллельный режим перекрывает стадии (предобработка, OCR, постобработка)
                для ускорения обработки. По умолчанию True.

        Returns:
            Кортеж (сгенерированный XML в формате <sheme>, путь к OCR логу).
        """
        start_time = time.time()
        video_path = Path(video_path)
        video_id = video_path.stem.split("_")[0] if "_" in video_path.stem else video_path.stem

        # Сохраняем callback для использования в _calibrate и других методах
        self._progress_callback = progress_callback

        # Инициализация OCR логгера
        if progress_callback:
            progress_callback(0, 0, "initializing")
        ocr_log = OcrLogger(
            video_id=video_id,
            config={
                "snapshot_interval_ms": settings.snapshot_interval_ms,
                "video_type": str(video_type) if video_type else "auto",
                "param_table": Path(param_table_path).name if param_table_path else "none",
            },
        )
        logger.info("OCR логгер инициализирован для видео: %s", video_id)

        # Загрузка таблицы параметров (опционально)
        if param_table_path is not None:
            param_table_path = Path(param_table_path)
            if progress_callback:
                progress_callback(0, 0, "loading_params")
            self._param_table = load_parameter_table(param_table_path)
            logger.info("Загружено %d параметров из %s", len(self._param_table), param_table_path.name)
        else:
            self._param_table = []
            logger.info("Таблица параметров не указана — используется OCR-only режим")

        # Сброс кэшей предобработки и зон для нового видео
        if progress_callback:
            progress_callback(0, 0, "preparing")
        from app.core.frame_preprocessor import clear_preprocess_cache
        from app.core.zone_segmentor import invalidate_zone_cache
        clear_preprocess_cache()
        invalidate_zone_cache()

        # Извлечение кадров: ленивый режим (экономия памяти ~4 ГБ)
        if progress_callback:
            progress_callback(0, 0, "frame_extraction")
        from app.core.video_ingestion import LazyFrameExtractor

        with LazyFrameExtractor(video_path, interval_ms=settings.snapshot_interval_ms) as frame_extractor:
            total_frames = frame_extractor.total_frames
            logger.info("Ленивое извлечение: ~%d кадров из %s", total_frames, video_path.name)

            # Определение типа видео (на первом кадре)
            first_frame = None
            first_timestamp = None
            for f, ts in frame_extractor:
                first_frame = f
                first_timestamp = ts
                break

            if first_frame is None:
                raise ValueError(f"Не удалось извлечь кадры из {video_path}")

            if video_type is None:
                from app.core.video_ingestion import detect_video_type
                if progress_callback:
                    progress_callback(0, total_frames, "detecting_video_type")
                video_type = detect_video_type(first_frame, video_path=str(video_path))
            logger.info("Тип видео: %s", video_type.value)

            # Сброс состояния конвейера
            self._value_processor.reset()
            self._tab_tracker.reset()
            self._snapshots = []
            self._prev_frame_gray = None
            self._prev_params = {}
            self._duplicate_skip_count = 0
            self._tab_change_count = 0
            self._current_sheet_name = ""
            self._adaptive_skip_interval = ADAPTIVE_SKIP_INITIAL
            self._consecutive_duplicates = 0
            self._frames_since_last_process = 0
            self._skip_stats = {
                "total_frames": 0,
                "processed_frames": 0,
                "skipped_by_adaptive": 0,
                "skipped_by_duplicate": 0,
                "interval_changes": 0,
                "interval_history": [],
            }

            # Этап 1: Калибровка (на первом кадре)
            self._calibrate(first_frame, video_type)

            if progress_callback:
                progress_callback(0, total_frames, "calibration")

            # Этап 2: Обработка всех кадров
            if use_parallel:
                logger.info("Используется оптимизированный параллельный конвейер")
                self._run_optimized_pipeline(
                    frame_extractor, video_type, total_frames, ocr_log,
                    progress_callback, video_id, first_frame, first_timestamp
                )
            else:
                # Последовательная обработка с ленивым извлечением
                logger.info("Используется последовательная обработка")
                self._run_sequential_lazy_pipeline(
                    frame_extractor, video_type, total_frames, ocr_log,
                    progress_callback, video_id, first_frame, first_timestamp
                )

        # Генерация XML
        xml = generate_xml(self._snapshots)

        # Сохраняем OCR лог
        log_path = ocr_log.save()
        logger.info("OCR лог сохранён: %s", log_path)

        elapsed = time.time() - start_time

        # Логируем статистику пропуска кадров
        self._log_skip_stats()

        logger.info(
            "Обработка завершена за %.1fс, %d снимков (пропущено дубликатов: %d, "
            "пропущено адаптивно: %d, обработано: %d/%d)",
            elapsed, len(self._snapshots), self._duplicate_skip_count,
            self._skip_stats["skipped_by_adaptive"],
            self._skip_stats["processed_frames"],
            self._skip_stats["total_frames"]
        )

        return xml, str(log_path)

    def _run_sequential_pipeline(
        self,
        frames: list[tuple[np.ndarray, float]],
        video_type: VideoType,
        total_frames: int,
        ocr_log: OcrLogger,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        """Последовательная обработка кадров (оригинальный алгоритм).

        Args:
            frames: Список (frame, timestamp_ms) для обработки.
            video_type: Тип видео.
            total_frames: Общее количество кадров.
            ocr_log: OCR логгер.
            progress_callback: Callback для отчёта о прогрессе.
        """
        # Импортируем metrics_collector для записи метрик
        from app.api.routes.pipeline import get_metrics_collector
        metrics_collector = get_metrics_collector()

        # Последовательная обработка
        prev_frame: np.ndarray | None = None
        error_count = 0
        max_errors = max(10, total_frames // 10)  # Допускаем до 10% ошибок

        for idx, (frame, timestamp) in enumerate(frames):
            self._skip_stats["total_frames"] += 1
            frame_start = time.perf_counter()
            ts_str = format_timestamp(timestamp / 1000.0) if isinstance(timestamp, (int, float)) else str(timestamp)

            # Начинаем запись кадра в лог
            ocr_log.start_frame(idx, ts_str)

            # === Adaptive Frame Skipping: проверяем нужно ли обрабатывать этот кадр ===
            self._frames_since_last_process += 1
            should_skip_adaptive = self._frames_since_last_process < self._adaptive_skip_interval

            if should_skip_adaptive and self._prev_params:
                # Пропускаем кадр по адаптивному интервалу
                self._skip_stats["skipped_by_adaptive"] += 1
                snapshot_params = self._prev_params.copy()
                self._duplicate_skip_count += 1

                # Создаём снимок с предыдущими значениями
                param_metadata = self._get_param_metadata(list(snapshot_params.keys()))
                snapshot = create_snapshot(timestamp, snapshot_params, param_metadata)
                self._snapshots.append(snapshot)

                # Отправляем WebSocket-сообщения (пропущенный кадр)
                from app.api.routes.pipeline import _ws_broadcast_sync
                for pid, val in snapshot_params.items():
                    meta = param_metadata.get(pid, ("", "", ""))
                    _ws_broadcast_sync({
                        "type": "ocr_result",
                        "video_id": "",
                        "param_id": pid,
                        "label": meta[0] or f"P{pid}",
                        "value": val,
                        "confidence": 0.85,
                        "source": "adaptive_skip",
                        "processing_ms": 0.05,
                        "timestamp": ts_str,
                        "short_name": meta[0],
                        "full_name": meta[1],
                        "unit": meta[2],
                    })

                # Логируем результат как пропущенный
                from app.core.ocr_models import RecognitionResult
                result = RecognitionResult(
                    raw_fields={pid: str(val) for pid, val in snapshot_params.items()},
                    confidence=0.85,
                    source="adaptive_skip",
                    pairs=[],
                    frame_idx=idx,
                    processing_ms=0.05,
                )
                ocr_log.log_result(result, is_duplicate=True, fallback=None)
                continue

            # Детекция активной вкладки SCADA
            tab_result = detect_active_scada_tab(frame, num_tabs=4)
            detected_tab = self._tab_tracker.update(tab_result)

            # Проверяем смену вкладки
            if has_tab_changed(self._get_tab_index_from_sheet(self._current_sheet_name), detected_tab):
                self._tab_change_count += 1
                prev_sheet = self._current_sheet_name
                new_sheet = self._get_sheet_name_from_tab(detected_tab)

                # Сохраняем текущую калибровку под старым именем вкладки
                # (или первичную калибровку если prev_sheet пуст)
                if prev_sheet and self._calibration:
                    self._tab_calibrations[prev_sheet] = self._calibration
                    logger.debug("Сохранена калибровка для вкладки '%s' (%d привязок)",
                                 prev_sheet, len(self._calibration.mappings))
                elif not prev_sheet and self._calibration:
                    # Первая детекция вкладки — сохраняем начальную калибровку
                    self._tab_calibrations[new_sheet] = self._calibration
                    logger.debug("Сохранена первичная калибровка под вкладкой '%s' (%d привязок)",
                                 new_sheet, len(self._calibration.mappings))

                # Обновляем имя текущей вкладки
                self._current_sheet_name = new_sheet
                self._frames_since_tab_change = 0  # Сброс счётчика кадров после смены вкладки

                logger.info(
                    "Смена вкладки SCADA: %s -> %s (вкладка %d)",
                    prev_sheet or "none",
                    self._current_sheet_name,
                    detected_tab
                )

                # Проверяем есть ли кэшированная калибровка для новой вкладки
                if new_sheet in self._tab_calibrations:
                    self._calibration = self._tab_calibrations[new_sheet]
                    logger.info("Восстановлена калибровка для вкладки '%s' (%d привязок)",
                                new_sheet, len(self._calibration.mappings))
                else:
                    # Новая вкладка — запускаем калибровку с Florence-2 grounding
                    logger.info("Новая вкладка '%s' — запуск калибровки", new_sheet)
                    try:
                        from app.core.florence_detector import FlorenceDetector

                        florence = self._get_florence_detector()
                        if florence is None:
                            florence = FlorenceDetector()
                            self._florence_detector = florence

                        new_mappings = calibrate_with_grounding(
                            frame, self._get_params_for_current_tab(), florence
                        )
                        if new_mappings:
                            new_profile = CalibrationProfile()
                            new_profile.mnemonic_name = new_sheet
                            for mapping in new_mappings:
                                new_profile.add_mapping(mapping)
                            self._calibration = new_profile
                            self._tab_calibrations[new_sheet] = new_profile
                            logger.info("Калибровка для '%s' завершена: %d привязок",
                                        new_sheet, len(new_mappings))
                        else:
                            # Fallback к существующему методу калибровки
                            self._calibrate(frame, video_type)
                            if self._calibration:
                                self._tab_calibrations[new_sheet] = self._calibration
                    except Exception as e:
                        logger.warning("Ошибка калибровки новой вкладки: %s — используем fallback", e)
                        self._calibrate(frame, video_type)
                        if self._calibration:
                            self._tab_calibrations[new_sheet] = self._calibration

                # Сбрасываем историю confidence scorer при смене вкладки
                if hasattr(self._confidence_scorer, 'reset_history'):
                    self._confidence_scorer.reset_history()

                # Сбрасываем кэш параметров и адаптивный интервал при смене вкладки
                self._prev_params = {}
                self._reset_adaptive_skip("tab_change")

            try:
                snapshot_params = self._process_single_frame(
                    idx, frame, timestamp, ts_str, video_type, prev_frame,
                    total_frames, ocr_log
                )
                # Обновляем prev_frame для следующей итерации
                prev_frame = frame
            except Exception as e:
                error_count += 1
                logger.error(
                    "Ошибка обработки кадра %d/%d: %s (ошибок: %d/%d)",
                    idx + 1, total_frames, e, error_count, max_errors
                )
                # Продолжаем со следующим кадром, используя предыдущие значения если есть
                if self._prev_params:
                    snapshot_params = self._prev_params.copy()
                    snapshot = create_snapshot(timestamp, snapshot_params)
                    self._snapshots.append(snapshot)
                else:
                    # Нет предыдущих значений — пропускаем кадр
                    frames[idx] = (None, timestamp)  # type: ignore[misc]
                    del frame
                    continue

                if error_count > max_errors:
                    logger.error(
                        "Слишком много ошибок (%d > %d), прерываем обработку",
                        error_count, max_errors
                    )
                    raise RuntimeError(
                        f"Превышен лимит ошибок при обработке: {error_count}"
                    ) from e

            # Обновление статуса и прогресса
            if idx % 10 == 0:
                progress = (idx + 1) / total_frames * 100
                logger.info("Обработано %d/%d кадров (%.1f%%)", idx + 1, total_frames, progress)

            # Вызов callback для обновления прогресса
            if progress_callback:
                progress_callback(idx + 1, total_frames, "ocr")

            # Записываем метрики для кадра (latency и параметры)
            frame_latency_ms = (time.perf_counter() - frame_start) * 1000
            param_results = {pid: (val, None) for pid, val in snapshot_params.items()}
            metrics_collector.record_frame(frame_latency_ms, param_results)

            # Отправляем WebSocket-сообщения для каждого параметра
            from app.api.routes.pipeline import _ws_broadcast_sync
            param_metadata = self._get_param_metadata(list(snapshot_params.keys()))
            for pid, val in snapshot_params.items():
                meta = param_metadata.get(pid, ("", "", ""))
                _ws_broadcast_sync({
                    "type": "ocr_result",
                    "video_id": "",  # Будет заполнено в callback
                    "param_id": pid,
                    "label": meta[0] or f"P{pid}",
                    "value": val,
                    "confidence": 0.85,  # Оценочное значение для pipeline режима
                    "source": "pipeline_roi",
                    "processing_ms": frame_latency_ms,
                    "timestamp": ts_str,
                    "short_name": meta[0],
                    "full_name": meta[1],
                    "unit": meta[2],
                })

            # Каждые 10 кадров отправляем метрики через WebSocket
            if idx % 10 == 0:
                metrics_summary = metrics_collector.get_summary()
                _ws_broadcast_sync({
                    "type": "metrics_update",
                    "video_id": "",
                    "metrics": metrics_summary,
                })

        # Логируем статистику ошибок
        if error_count > 0:
            logger.warning("Последовательная обработка: %d ошибок из %d кадров", error_count, total_frames)

    def _reset_adaptive_skip(self, reason: str) -> None:
        """Сбрасывает адаптивный интервал пропуска к начальному значению.

        Args:
            reason: Причина сброса (для логирования).
        """
        old_interval = self._adaptive_skip_interval
        self._adaptive_skip_interval = ADAPTIVE_SKIP_INITIAL
        self._consecutive_duplicates = 0
        self._frames_since_last_process = 0

        if old_interval != ADAPTIVE_SKIP_INITIAL:
            self._skip_stats["interval_changes"] += 1
            self._skip_stats["interval_history"].append({
                "action": "reset",
                "from": old_interval,
                "to": ADAPTIVE_SKIP_INITIAL,
                "reason": reason,
            })
            logger.info(
                "Адаптивный интервал сброшен: %d -> %d (причина: %s)",
                old_interval, ADAPTIVE_SKIP_INITIAL, reason
            )

    def _increase_adaptive_skip(self) -> None:
        """Увеличивает адаптивный интервал пропуска кадров."""
        old_interval = self._adaptive_skip_interval
        new_interval = min(
            old_interval * ADAPTIVE_SKIP_MULTIPLIER,
            ADAPTIVE_SKIP_MAX
        )

        if new_interval != old_interval:
            self._adaptive_skip_interval = int(new_interval)
            self._skip_stats["interval_changes"] += 1
            self._skip_stats["interval_history"].append({
                "action": "increase",
                "from": old_interval,
                "to": self._adaptive_skip_interval,
                "consecutive_duplicates": self._consecutive_duplicates,
            })
            logger.info(
                "Адаптивный интервал увеличен: %d -> %d (последовательных дубликатов: %d)",
                old_interval, self._adaptive_skip_interval, self._consecutive_duplicates
            )

    def _log_skip_stats(self) -> None:
        """Логирует детальную статистику пропуска кадров."""
        stats = self._skip_stats
        total = stats["total_frames"]
        processed = stats["processed_frames"]
        skipped_adaptive = stats["skipped_by_adaptive"]
        skipped_duplicate = stats["skipped_by_duplicate"]
        total_skipped = skipped_adaptive + skipped_duplicate

        if total > 0:
            skip_rate = (total_skipped / total) * 100
            adaptive_rate = (skipped_adaptive / total) * 100
            duplicate_rate = (skipped_duplicate / total) * 100

            logger.info(
                "=== Статистика пропуска кадров ===\n"
                "  Всего кадров: %d\n"
                "  Обработано: %d (%.1f%%)\n"
                "  Пропущено (адаптивно): %d (%.1f%%)\n"
                "  Пропущено (дубликаты): %d (%.1f%%)\n"
                "  Общий skip rate: %.1f%%\n"
                "  Изменений интервала: %d",
                total, processed, (processed / total) * 100,
                skipped_adaptive, adaptive_rate,
                skipped_duplicate, duplicate_rate,
                skip_rate, stats["interval_changes"]
            )

            if stats["interval_history"]:
                logger.debug("История изменений интервала: %s", stats["interval_history"])

    def _run_sequential_lazy_pipeline(
        self,
        frame_extractor,
        video_type: VideoType,
        total_frames: int,
        ocr_log: OcrLogger,
        progress_callback: Callable[[int, int, str], None] | None = None,
        video_id: str = "",
        first_frame: np.ndarray | None = None,
        first_timestamp: float | None = None,
    ) -> None:
        """Последовательная обработка с ленивым извлечением кадров.

        Отличие от _run_sequential_pipeline: кадры извлекаются по одному
        (лениво), а не загружаются все в память. Та же точность.

        Args:
            frame_extractor: LazyFrameExtractor для ленивого чтения кадров.
            video_type: Тип видео.
            total_frames: Общее количество кадров.
            ocr_log: OCR логгер.
            progress_callback: Callback для отчёта о прогрессе.
            video_id: ID видео для кэширования.
            first_frame: Первый кадр (уже извлечён для калибровки).
            first_timestamp: Таймстемп первого кадра.
        """
        from app.api.routes.pipeline import get_metrics_collector, _ws_broadcast_sync
        from app.core.zone_segmentor import invalidate_zone_cache
        metrics_collector = get_metrics_collector()

        prev_frame: np.ndarray | None = None
        error_count = 0
        max_errors = max(10, total_frames // 10)
        idx = 0

        # Обрабатываем первый кадр (уже извлечён)
        if first_frame is not None and first_timestamp is not None:
            ts_str = format_timestamp(first_timestamp / 1000.0) if isinstance(first_timestamp, (int, float)) else str(first_timestamp)
            ocr_log.start_frame(0, ts_str)
            try:
                self._process_single_frame(
                    0, first_frame, first_timestamp, ts_str, video_type,
                    None, total_frames, ocr_log
                )
                prev_frame = first_frame
            except Exception as e:
                logger.error("Ошибка обработки первого кадра: %s", e)
            idx = 1

        # Обрабатываем оставшиеся кадры лениво
        for frame, timestamp in frame_extractor:
            self._skip_stats["total_frames"] += 1
            frame_start = time.perf_counter()
            ts_str = format_timestamp(timestamp / 1000.0) if isinstance(timestamp, (int, float)) else str(timestamp)

            ocr_log.start_frame(idx, ts_str)

            # Адаптивный пропуск
            self._frames_since_last_process += 1
            should_skip_adaptive = self._frames_since_last_process < self._adaptive_skip_interval

            if should_skip_adaptive and self._prev_params:
                self._skip_stats["skipped_by_adaptive"] += 1
                snapshot_params = self._prev_params.copy()
                self._duplicate_skip_count += 1

                param_metadata = self._get_param_metadata(list(snapshot_params.keys()))
                snapshot = create_snapshot(timestamp, snapshot_params, param_metadata)
                self._snapshots.append(snapshot)

                _ws_broadcast_sync({
                    "type": "ocr_result",
                    "video_id": video_id,
                    "param_id": 0,
                    "label": "skip",
                    "value": "",
                    "confidence": 0.85,
                    "source": "adaptive_skip",
                    "processing_ms": 0.05,
                    "timestamp": ts_str,
                })
                idx += 1
                prev_frame = frame
                del frame  # Освобождаем память
                continue

            # Дедупликация
            is_duplicate, is_scene_change = self._is_duplicate_frame(frame)

            if is_scene_change:
                logger.info("Смена сцены на кадре %d", idx + 1)
                self._reset_adaptive_skip("scene_change")
                invalidate_zone_cache()  # Инвалидируем кэш зон

            if is_duplicate and self._prev_params and not is_scene_change:
                self._duplicate_skip_count += 1
                self._skip_stats["skipped_by_duplicate"] += 1
                self._consecutive_duplicates += 1
                self._consecutive_duplicate_count += 1  # Новый счётчик
                self._frames_since_tab_change += 1  # Инкремент после обработки
                if self._consecutive_duplicates >= ADAPTIVE_SKIP_DUPLICATE_THRESHOLD:
                    self._increase_adaptive_skip()
                    self._consecutive_duplicates = 0
                snapshot_params = self._prev_params.copy()
                param_metadata = self._get_param_metadata(list(snapshot_params.keys()))
                snapshot = create_snapshot(timestamp, snapshot_params, param_metadata)
                self._snapshots.append(snapshot)
                idx += 1
                prev_frame = frame
                del frame
                continue

            if not is_duplicate:
                self._consecutive_duplicates = 0
                self._consecutive_duplicate_count = 0  # Сброс нового счётчика
                self._last_processed_timestamp = time.time()  # Обновление времени
                if self._adaptive_skip_interval > ADAPTIVE_SKIP_INITIAL:
                    self._reset_adaptive_skip("unique_frame_detected")

            try:
                snapshot_params = self._process_single_frame(
                    idx, frame, timestamp, ts_str, video_type, prev_frame,
                    total_frames, ocr_log
                )
                prev_frame = frame

                frame_latency_ms = (time.perf_counter() - frame_start) * 1000
                param_results = {pid: (val, None) for pid, val in snapshot_params.items()}
                metrics_collector.record_frame(frame_latency_ms, param_results)

            except Exception as e:
                error_count += 1
                logger.error("Ошибка обработки кадра %d: %s", idx, e)
                if self._prev_params:
                    snapshot_params = self._prev_params.copy()
                    snapshot = create_snapshot(timestamp, snapshot_params)
                    self._snapshots.append(snapshot)

            if idx % 10 == 0:
                progress = (idx + 1) / max(total_frames, 1) * 100
                logger.info("Обработано %d/~%d кадров (%.1f%%)", idx + 1, total_frames, progress)

            if progress_callback:
                progress_callback(idx + 1, total_frames, "ocr")

            idx += 1
            del frame  # Освобождаем память немедленно

    def _run_optimized_pipeline(
        self,
        frame_extractor,
        video_type: VideoType,
        total_frames: int,
        ocr_log: OcrLogger,
        progress_callback: Callable[[int, int, str], None] | None = None,
        video_id: str = "",
        first_frame: np.ndarray | None = None,
        first_timestamp: float | None = None,
    ) -> None:
        """Оптимизированный параллельный конвейер с Enhanced OCR и ленивым извлечением.

        Ключевые оптимизации (без потери качества):
        1. Ленивое извлечение кадров — кадры загружаются по одному
        2. Кэширование зон — segment_zones() вызывается 1 раз (не на каждый кадр)
        3. Enhanced pipeline — PaddleOCR + Florence + Layout Analysis
        4. Предобработка в отдельном потоке — перекрытие с OCR
        5. Инвалидация кэша зон при смене сцены

        Точность: ИДЕНТИЧНА последовательному конвейеру — те же алгоритмы,
        те же параметры OCR, та же фильтрация.

        Архитектура:
        - Главный поток: ленивое извлечение + дедупликация + предобработка
        - OCR-поток: распознавание (PaddleOCR singleton, thread-safe lock)
        - Постобработка-потоки: маппинг параметров, генерация XML-снимков

        Args:
            frame_extractor: LazyFrameExtractor для ленивого чтения кадров.
            video_type: Тип видео.
            total_frames: Общее количество кадров.
            ocr_log: OCR логгер.
            progress_callback: Callback для отчёта о прогрессе.
            video_id: ID видео для кэширования.
            first_frame: Первый кадр (уже извлечён для калибровки).
            first_timestamp: Таймстемп первого кадра.
        """
        from app.api.routes.pipeline import get_metrics_collector, _ws_broadcast_sync
        from app.core.zone_segmentor import invalidate_zone_cache
        metrics_collector = get_metrics_collector()

        pipeline_start = time.perf_counter()

        # Инициализируем Florence-2 для основного пути обработки
        try:
            self._get_florence_detector()
            if self._florence_detector is not None:
                logger.info("Florence-2 инициализирован для оптимизированного пайплайна")
        except Exception as e:
            logger.warning("Не удалось инициализировать Florence-2: %s — используется fallback", e)

        # Очереди между стадиями (увеличенные размеры для параллельной предобработки)
        preprocess_queue: queue.Queue = queue.Queue(maxsize=20)
        ocr_queue: queue.Queue = queue.Queue(maxsize=16)
        result_queue: queue.Queue = queue.Queue(maxsize=32)

        # Событие остановки
        stop_event = threading.Event()

        # Статистика производительности
        stage_timings: dict[str, list[float]] = {
            "preprocess": [],
            "ocr": [],
            "postprocess": [],
        }
        stage_timings_lock = threading.Lock()

        # Буфер упорядочивания результатов (для сохранения хронологического порядка)
        results_lock = threading.Lock()
        results_by_idx: dict[int, dict] = {}
        next_output_idx = [0]

        # Оценка количества кадров для прогресса
        processed_count = 0
        duplicate_count = 0

        def _flush_ordered_results() -> None:
            """Сбрасывает накопленные результаты в хронологическом порядке.

            Вызывается под results_lock. Применяет временное сглаживание
            и сохраняет снимки в правильном порядке.
            """
            while next_output_idx[0] in results_by_idx:
                data = results_by_idx.pop(next_output_idx[0])
                self._prev_params = data.get('params', {})
                self._snapshots.append(data['snapshot'])
                next_output_idx[0] += 1

        def _preprocess_worker() -> None:
            """Рабочий поток предобработки кадров (CPU-bound).

            Выполняет CLAHE, коррекцию перспективы, сегментацию зон.
            Результат отправляет в OCR-очередь.
            """
            while not stop_event.is_set():
                try:
                    item = preprocess_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is _SENTINEL:
                    break

                idx, frame, timestamp, video_type_str, ts_str = item

                try:
                    preprocess_t0 = time.perf_counter()
                    processed, _scale = preprocess_frame(
                        frame, video_type, None, self._screen_corners, video_id
                    )
                    preprocess_ms = (time.perf_counter() - preprocess_t0) * 1000

                    with stage_timings_lock:
                        stage_timings["preprocess"].append(preprocess_ms)

                    # Сегментация зон (пропускаем если Florence активен)
                    if self._florence_detector is None or self._frames_since_tab_change < FORCE_PROCESS_AFTER_TAB_CHANGE:
                        zones = segment_zones(processed)
                    else:
                        zones = []

                    ocr_queue.put((idx, frame, timestamp, processed, zones, ts_str))
                except Exception as e:
                    logger.error("Preprocess error frame %d: %s", idx, e)

        def _ocr_worker() -> None:
            """Рабочий поток OCR (единый для thread-safety PaddleOCR).

            Использует batch-ROI OCR когда есть калибровка — один вызов
            PaddleOCR для всех ROI вместо N отдельных вызовов.
            """
            while not stop_event.is_set():
                try:
                    item = ocr_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is _SENTINEL:
                    result_queue.put(_SENTINEL)
                    break

                idx, frame, timestamp, processed, zones, ts_str = item

                try:
                    ocr_t0 = time.perf_counter()
                    # Используем batch ROI OCR если есть калибровка с ROI
                    snapshot_params = self._run_ocr_on_preprocessed(processed, zones, ocr_log)
                    ocr_ms = (time.perf_counter() - ocr_t0) * 1000

                    with stage_timings_lock:
                        stage_timings["ocr"].append(ocr_ms)

                    result_queue.put((idx, frame, timestamp, processed, snapshot_params, ts_str, ocr_ms))
                except Exception as e:
                    logger.error("OCR error frame %d: %s", idx, e)
                    result_queue.put((idx, frame, timestamp, processed, None, ts_str, 0.0))

        def _postprocess_worker() -> None:
            """Рабочий поток постобработки результатов OCR.

            Использует буфер упорядочивания для сохранения хронологического
            порядка снимков при параллельной обработке.
            """
            while not stop_event.is_set():
                try:
                    item = result_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is _SENTINEL:
                    break

                idx, frame, timestamp, processed, snapshot_params, ts_str, ocr_ms = item

                if snapshot_params is None:
                    continue

                try:
                    post_t0 = time.perf_counter()

                    param_metadata = self._get_param_metadata(list(snapshot_params.keys()))
                    snapshot = create_snapshot(timestamp, snapshot_params, param_metadata)

                    # Сохраняем в буфер упорядочивания
                    with results_lock:
                        results_by_idx[idx] = {
                            'params': snapshot_params.copy(),
                            'snapshot': snapshot,
                        }
                        _flush_ordered_results()

                    from app.core.ocr_models import RecognitionResult
                    result = RecognitionResult(
                        raw_fields={pid: str(val) for pid, val in snapshot_params.items()},
                        confidence=0.85,
                        source="optimized_pipeline",
                        pairs=[],
                        frame_idx=idx,
                        processing_ms=ocr_ms,
                    )
                    ocr_log.log_result(result, is_duplicate=False, fallback=None)

                    post_ms = (time.perf_counter() - post_t0) * 1000
                    with stage_timings_lock:
                        stage_timings["postprocess"].append(post_ms)

                    frame_latency_ms = ocr_ms + post_ms
                    param_results = {pid: (val, None) for pid, val in snapshot_params.items()}
                    metrics_collector.record_frame(frame_latency_ms, param_results)

                    # WebSocket обновления (батч — отправляем один раз на кадр)
                    _ws_broadcast_sync({
                        "type": "ocr_result",
                        "video_id": video_id,
                        "param_id": 0,
                        "label": "batch",
                        "value": f"{len(snapshot_params)} params",
                        "confidence": 0.85,
                        "source": "optimized_pipeline",
                        "processing_ms": frame_latency_ms,
                        "timestamp": ts_str,
                    })

                    if idx % 10 == 0:
                        metrics_summary = metrics_collector.get_summary()
                        _ws_broadcast_sync({
                            "type": "metrics_update",
                            "video_id": video_id,
                            "metrics": metrics_summary,
                        })

                except Exception as e:
                    logger.error("Postprocess error frame %d: %s", idx, e)

        # Запускаем рабочие потоки
        num_post_workers = settings.postprocess_workers
        logger.info("Параллельная предобработка: %d worker(s) запущено", PREPROCESS_WORKERS)

        preprocess_threads = [
            threading.Thread(target=_preprocess_worker, daemon=True, name=f"PreProcess-{i}")
            for i in range(PREPROCESS_WORKERS)
        ]
        ocr_thread = threading.Thread(target=_ocr_worker, daemon=True, name="OCR-Worker-Opt")
        post_threads = [
            threading.Thread(target=_postprocess_worker, daemon=True, name=f"PostProcess-{i}")
            for i in range(num_post_workers)
        ]

        for t in preprocess_threads:
            t.start()
        ocr_thread.start()
        for t in post_threads:
            t.start()

        # Главный поток: ленивое извлечение + предобработка + дедупликация
        prev_frame: np.ndarray | None = None

        # Обрабатываем первый кадр (уже извлечён для калибровки)
        idx = 0
        if first_frame is not None and first_timestamp is not None:
            ts_str = format_timestamp(first_timestamp / 1000.0) if isinstance(first_timestamp, (int, float)) else str(first_timestamp)
            ocr_log.start_frame(0, ts_str)

            # Предобработка первого кадра
            preprocess_t0 = time.perf_counter()
            processed, _scale = preprocess_frame(first_frame, video_type, None, self._screen_corners, video_id)
            preprocess_ms = (time.perf_counter() - preprocess_t0) * 1000
            ocr_log.log_stage("preprocess", preprocess_ms)
            with stage_timings_lock:
                stage_timings["preprocess"].append(preprocess_ms)

            # Сегментация зон для первого кадра (пропускаем если Florence активен)
            if self._florence_detector is None:
                zones = segment_zones(processed)
            else:
                zones = []
                logger.debug("Florence-путь: пропускаем сегментацию зон для первого кадра")

            ocr_queue.put((0, first_frame, first_timestamp, processed, zones, ts_str))
            prev_frame = first_frame
            processed_count += 1
            idx = 1

        # Обрабатываем оставшиеся кадры лениво
        for frame, timestamp in frame_extractor:
            self._skip_stats["total_frames"] += 1
            frame_start = time.perf_counter()
            ts_str = format_timestamp(timestamp / 1000.0) if isinstance(timestamp, (int, float)) else str(timestamp)

            ocr_log.start_frame(idx, ts_str)

            # Адаптивный пропуск
            self._frames_since_last_process += 1
            should_skip_adaptive = self._frames_since_last_process < self._adaptive_skip_interval

            if should_skip_adaptive and self._prev_params:
                self._skip_stats["skipped_by_adaptive"] += 1
                snapshot_params = self._prev_params.copy()
                self._duplicate_skip_count += 1

                param_metadata = self._get_param_metadata(list(snapshot_params.keys()))
                snapshot = create_snapshot(timestamp, snapshot_params, param_metadata)
                self._snapshots.append(snapshot)

                from app.core.ocr_models import RecognitionResult
                result = RecognitionResult(
                    raw_fields={pid: str(val) for pid, val in snapshot_params.items()},
                    confidence=0.85,
                    source="adaptive_skip",
                    pairs=[],
                    frame_idx=idx,
                    processing_ms=0.05,
                )
                ocr_log.log_result(result, is_duplicate=True, fallback=None)
                prev_frame = frame
                del frame  # Освобождаем память
                idx += 1
                continue

            # Дедупликация
            is_duplicate, is_scene_change = self._is_duplicate_frame(frame)

            if is_scene_change:
                logger.info("Смена сцены на кадре %d — инвалидируем кэш зон", idx + 1)
                self._reset_adaptive_skip("scene_change")
                invalidate_zone_cache()  # Ключевая оптимизация: сброс кэша зон

            # Florence-based tab detection в оптимизированном пайплайне
            if self._florence_detector is not None and is_scene_change:
                try:
                    scene_description = self._florence_detector.describe_scene(frame)
                    new_tab = scene_description.get("active_tab")
                    new_gpa_type = scene_description.get("gpa_type")

                    if new_tab is not None and new_tab != self._current_tab:
                        logger.info(
                            "Florence (optimized pipeline): смена вкладки '%s' -> '%s'",
                            self._current_tab, new_tab
                        )
                        self._current_tab = new_tab
                        self._frames_since_tab_change = 0
                        invalidate_zone_cache()

                        # Триггерим перекалибровку если есть параметры для новой вкладки
                        if self._param_table:
                            try:
                                new_mappings = calibrate_with_grounding(
                                    frame, self._get_params_for_current_tab(), self._florence_detector
                                )
                                if new_mappings:
                                    new_profile = CalibrationProfile()
                                    new_profile.mnemonic_name = new_tab
                                    for mapping in new_mappings:
                                        new_profile.add_mapping(mapping)
                                    self._calibration = new_profile
                                    self._tab_calibrations[new_tab] = new_profile
                                    logger.info(
                                        "Florence (optimized): калибровка для '%s' завершена: %d привязок",
                                        new_tab, len(new_mappings)
                                    )
                            except Exception as e:
                                logger.warning("Ошибка калибровки при смене вкладки: %s", e)

                    if new_gpa_type is not None:
                        self._current_gpa_type = new_gpa_type
                except Exception as e:
                    logger.warning("Ошибка Florence scene detection в optimized pipeline: %s", e)

            if is_duplicate and self._prev_params and not is_scene_change:
                duplicate_count += 1
                self._duplicate_skip_count += 1
                self._skip_stats["skipped_by_duplicate"] += 1
                self._consecutive_duplicates += 1
                self._consecutive_duplicate_count += 1  # Новый счётчик
                self._frames_since_tab_change += 1  # Инкремент после обработки

                if self._consecutive_duplicates >= ADAPTIVE_SKIP_DUPLICATE_THRESHOLD:
                    self._increase_adaptive_skip()
                    self._consecutive_duplicates = 0

                snapshot_params = self._prev_params.copy()
                param_metadata = self._get_param_metadata(list(snapshot_params.keys()))
                snapshot = create_snapshot(timestamp, snapshot_params, param_metadata)
                self._snapshots.append(snapshot)

                from app.core.ocr_models import RecognitionResult
                result = RecognitionResult(
                    raw_fields={pid: str(val) for pid, val in snapshot_params.items()},
                    confidence=0.85,
                    source="pipeline_duplicate",
                    pairs=[],
                    frame_idx=idx,
                    processing_ms=0.1,
                )
                ocr_log.log_result(result, is_duplicate=True, fallback=None)
                prev_frame = frame
                del frame
                idx += 1
                continue

            # Сброс счётчика дубликатов при уникальном кадре
            if not is_duplicate:
                self._consecutive_duplicates = 0
                self._consecutive_duplicate_count = 0  # Сброс нового счётчика
                self._last_processed_timestamp = time.time()  # Обновление времени
                if self._adaptive_skip_interval > ADAPTIVE_SKIP_INITIAL:
                    self._reset_adaptive_skip("unique_frame_detected")

            # Отправляем в очередь предобработки (параллельная обработка)
            self._skip_stats["processed_frames"] += 1
            self._frames_since_last_process = 0
            self._frames_since_tab_change += 1  # Инкремент после обработки
            processed_count += 1

            video_type_str = video_type.value if isinstance(video_type, VideoType) else str(video_type)
            preprocess_queue.put((idx, frame, timestamp, video_type_str, ts_str))
            prev_frame = frame
            # НЕ del frame — preprocess-поток ещё может использовать его

            # Прогресс
            if idx % 10 == 0:
                progress = (idx + 1) / max(total_frames, 1) * 100
                logger.info("Обработано %d/~%d кадров (%.1f%%)", idx + 1, total_frames, progress)
            if progress_callback:
                progress_callback(idx + 1, total_frames, "ocr")

            idx += 1

        # Сигнал завершения для предобработки
        for _ in preprocess_threads:
            preprocess_queue.put(_SENTINEL)

        # Ожидаем завершение предобработки
        for t in preprocess_threads:
            t.join(timeout=60.0)

        # Сигнал завершения для OCR
        ocr_queue.put(_SENTINEL)

        # Ожидаем завершение OCR-потока
        ocr_thread.join(timeout=300.0)

        # Сигнал завершения для постобработки
        for _ in post_threads:
            result_queue.put(_SENTINEL)

        # Ожидаем завершение постобработки
        for t in post_threads:
            t.join(timeout=60.0)

        # Финальный сброс накопленных результатов
        with results_lock:
            _flush_ordered_results()

        pipeline_elapsed = time.perf_counter() - pipeline_start

        # Логируем статистику
        logger.info(
            "Optimized pipeline: %d total frames, %d processed, %d duplicates, %.1fs total, "
            "%.2fs/frame avg",
            idx, processed_count, duplicate_count, pipeline_elapsed,
            pipeline_elapsed / max(1, processed_count)
        )

        with stage_timings_lock:
            if stage_timings["preprocess"]:
                avg_preprocess = sum(stage_timings["preprocess"]) / len(stage_timings["preprocess"])
                logger.info("Stage timings: preprocess=%.1fms avg", avg_preprocess)
            if stage_timings["ocr"]:
                avg_ocr = sum(stage_timings["ocr"]) / len(stage_timings["ocr"])
                logger.info("Stage timings: ocr=%.1fms avg", avg_ocr)
            if stage_timings["postprocess"]:
                avg_post = sum(stage_timings["postprocess"]) / len(stage_timings["postprocess"])
                logger.info("Stage timings: postprocess=%.1fms avg", avg_post)

    def _process_single_frame(
        self,
        idx: int,
        frame: np.ndarray,
        timestamp: str,
        ts_str: str,
        video_type: VideoType,
        prev_frame: np.ndarray | None,
        total_frames: int,
        ocr_log: OcrLogger,
    ) -> dict[int, str]:
        """Обрабатывает один кадр с возможностью восстановления при ошибках.

        Args:
            idx: Индекс кадра.
            frame: Кадр для обработки.
            timestamp: Таймстемп кадра.
            ts_str: Строковое представление таймстемпа.
            video_type: Тип видео.
            prev_frame: Предыдущий кадр.
            total_frames: Общее количество кадров.
            ocr_log: OCR логгер.

        Returns:
            Словарь распознанных параметров.

        Raises:
            Exception: При ошибке обработки кадра.
        """
        frame_start = time.perf_counter()

        # Обновляем статистику обработанных кадров
        self._skip_stats["processed_frames"] += 1
        self._frames_since_last_process = 0

        # Проверка на дубликат кадра (быстрая, до предобработки)
        is_duplicate, is_scene_change = self._is_duplicate_frame(frame)

        if is_scene_change:
            # Смена сцены — сбрасываем кэш параметров и адаптивный интервал
            logger.info("Смена сцены на кадре %d — принудительная обработка", idx + 1)
            self._reset_adaptive_skip("scene_change")

        if is_duplicate and self._prev_params and not is_scene_change:
            # Пропускаем обработку — используем значения из предыдущего кадра
            snapshot_params = self._prev_params.copy()
            self._duplicate_skip_count += 1
            self._skip_stats["skipped_by_duplicate"] += 1

            # Обновляем счётчик последовательных дубликатов
            self._consecutive_duplicates += 1

            # Увеличиваем адаптивный интервал если достигнут порог
            if self._consecutive_duplicates >= ADAPTIVE_SKIP_DUPLICATE_THRESHOLD:
                self._increase_adaptive_skip()
                self._consecutive_duplicates = 0

            # Создаём снимок с метаданными
            param_metadata = self._get_param_metadata(list(snapshot_params.keys()))
            snapshot = create_snapshot(timestamp, snapshot_params, param_metadata)
            self._snapshots.append(snapshot)

            # Отправляем WebSocket-сообщения для каждого параметра (дубликат)
            from app.api.routes.pipeline import _ws_broadcast_sync
            for pid, val in snapshot_params.items():
                meta = param_metadata.get(pid, ("", "", ""))
                _ws_broadcast_sync({
                    "type": "ocr_result",
                    "video_id": "",
                    "param_id": pid,
                    "label": meta[0] or f"P{pid}",
                    "value": val,
                    "confidence": 0.85,
                    "source": "pipeline_roi_duplicate",
                    "processing_ms": 0.1,
                    "timestamp": ts_str,
                    "short_name": meta[0],
                    "full_name": meta[1],
                    "unit": meta[2],
                })

            # Логируем результат как дубликат
            from app.core.ocr_models import RecognitionResult
            result = RecognitionResult(
                raw_fields={pid: str(val) for pid, val in snapshot_params.items()},
                confidence=0.85,
                source="pipeline_roi_duplicate",
                pairs=[],
                frame_idx=idx,
                processing_ms=0.1,  # практически мгновенно
            )
            ocr_log.log_result(result, is_duplicate=True, fallback=None)

            latency_ms = (time.perf_counter() - frame_start) * 1000
            logger.info(
                "Кадр %d/%d: ДУБЛИКАТ (пропущен), %.0f мс",
                idx + 1, total_frames, latency_ms
            )

            # Инкрементируем счётчик последовательных дубликатов
            self._consecutive_duplicate_count += 1
            # Инкрементируем счётчик кадров после обработки
            self._frames_since_tab_change += 1

            return snapshot_params

        # Сбрасываем счётчик дубликатов при обработке уникального кадра
        if not is_duplicate:
            if self._consecutive_duplicates > 0:
                logger.debug(
                    "Сброс счётчика дубликатов (%d) — обработан уникальный кадр",
                    self._consecutive_duplicates
                )
            self._consecutive_duplicates = 0
            self._consecutive_duplicate_count = 0  # Сброс нового счётчика
            self._last_processed_timestamp = time.time()  # Обновление времени последней обработки
            # Сбрасываем адаптивный интервал при значимом изменении
            if self._adaptive_skip_interval > ADAPTIVE_SKIP_INITIAL:
                self._reset_adaptive_skip("unique_frame_detected")

        # === a) Детекция смены вкладки SCADA (BEFORE zone segmentation) ===
        # PRIMARY: Florence-based tab detection (если доступен)
        florence_tab_changed = False
        scene_description = None
        if self._florence_detector is not None:
            try:
                scene_t0 = time.perf_counter()
                scene_description = self._florence_detector.describe_scene(frame)
                scene_ms = (time.perf_counter() - scene_t0) * 1000
                ocr_log.log_stage("florence_scene", scene_ms)

                # Проверяем смену вкладки через Florence
                new_tab = scene_description.get("active_tab")
                new_gpa_type = scene_description.get("gpa_type")

                if new_tab is not None and new_tab != self._current_tab:
                    logger.info(
                        "Florence: смена вкладки '%s' -> '%s'",
                        self._current_tab, new_tab
                    )
                    self._current_tab = new_tab
                    florence_tab_changed = True

                # Обновляем тип ГПА
                if new_gpa_type is not None:
                    if new_gpa_type != self._current_gpa_type:
                        logger.info(
                            "Florence: изменение типа ГПА '%s' -> '%s'",
                            self._current_gpa_type, new_gpa_type
                        )
                    self._current_gpa_type = new_gpa_type

                # Проверяем наличие popup
                if scene_description.get("has_popup"):
                    logger.info("Florence: обнаружен popup — пропуск извлечения параметров")
                    if self._prev_params:
                        return self._prev_params.copy()
                    return {}

                # Проверяем фазу переключения вкладки (пустая схема)
                if (self._current_tab is not None and
                    new_tab is None and
                    self._current_gpa_type is not None and
                    not scene_description.get("description")):
                    logger.debug("Florence: фаза переключения вкладки — возвращаем предыдущие значения")
                    if self._prev_params:
                        return self._prev_params.copy()
                    return {}

            except Exception as e:
                logger.warning("Ошибка Florence scene detection: %s", e)
                scene_description = None

        # FALLBACK: Существующий ScadaTabTracker если Florence не сработал
        try:
            tab_t0 = time.perf_counter()
            tab_changed = self._scada_tab_tracker.check_tab_change(frame)
            if tab_changed and not florence_tab_changed:
                logger.info("Tab change detected (fallback) — инвалидация кэша зон")
                self._frames_since_tab_change = 0
                invalidate_zone_cache()
                increment_cache_epoch()
            tab_ms = (time.perf_counter() - tab_t0) * 1000
            ocr_log.log_stage("tab_detection", tab_ms)
        except Exception as e:
            logger.warning("Ошибка детекции вкладки: %s", e)

        # При смене вкладки через Florence — сбрасываем состояние
        if florence_tab_changed:
            self._frames_since_tab_change = 0
            invalidate_zone_cache()
            increment_cache_epoch()
            # Сбрасываем предыдущие параметры (новая вкладка = новая схема)
            self._prev_params = {}
            # Сбрасываем историю confidence scorer
            if hasattr(self._confidence_scorer, 'reset_history'):
                self._confidence_scorer.reset_history()

        # Предобработка
        preprocess_t0 = time.perf_counter()
        processed, _scale = preprocess_frame(
            frame, video_type, prev_frame, self._screen_corners
        )
        preprocess_ms = (time.perf_counter() - preprocess_t0) * 1000
        ocr_log.log_stage("preprocess", preprocess_ms)

        # === Mode A: Force full calibration on first 3 frames after tab change ===
        if self._frames_since_tab_change < FORCE_PROCESS_AFTER_TAB_CHANGE:
            if self._florence_detector is not None and self._param_table:
                try:
                    logger.info(
                        "Mode A: полная калибровка кадра %d после смены вкладки",
                        self._frames_since_tab_change
                    )
                    mode_a_t0 = time.perf_counter()
                    new_mappings = calibrate_with_grounding(
                        processed, self._get_params_for_current_tab(), self._florence_detector
                    )
                    if new_mappings:
                        new_profile = CalibrationProfile()
                        new_profile.mnemonic_name = self._current_tab or "unknown"
                        for mapping in new_mappings:
                            new_profile.add_mapping(mapping)
                        self._calibration = new_profile
                        logger.info("Mode A: калибровка завершена, %d привязок", len(new_mappings))
                    mode_a_ms = (time.perf_counter() - mode_a_t0) * 1000
                    ocr_log.log_stage("mode_a_calibration", mode_a_ms)
                except Exception as e:
                    logger.warning("Ошибка Mode A калибровки: %s", e)

        # Сегментация зон (пропускаем если Florence активен и работает)
        zones = []
        if self._florence_detector is None or self._frames_since_tab_change < FORCE_PROCESS_AFTER_TAB_CHANGE:
            zone_t0 = time.perf_counter()
            zones = segment_zones(processed)
            zone_ms = (time.perf_counter() - zone_t0) * 1000
            ocr_log.log_stage("zone_segmentation", zone_ms)
        else:
            # Bypass zone segmentation when Florence is primary
            ocr_log.log_stage("zone_segmentation", 0.0)

        # === b) Центральная зона: обрезка для OCR (только если нет Florence) ===
        central_zone = None
        crop_offset = (0, 0)  # (x_offset, y_offset)
        frame_for_ocr = processed
        if self._florence_detector is None and zones:
            central_zone = next(
                (z for z in zones if z.zone == ZoneType.CENTRAL_SCHEMA), None
            )
            if central_zone is not None:
                try:
                    crop_t0 = time.perf_counter()
                    h, w = processed.shape[:2]
                    # Извлекаем центральную зону
                    central_crop = extract_zone(processed, central_zone.bbox)
                    # Вычисляем смещение для обратного маппинга координат
                    x_offset = int(central_zone.bbox.x1 * w)
                    y_offset = int(central_zone.bbox.y1 * h)
                    crop_offset = (x_offset, y_offset)
                    frame_for_ocr = central_crop
                    crop_ms = (time.perf_counter() - crop_t0) * 1000
                    ocr_log.log_stage("central_crop", crop_ms)
                    logger.debug(
                        "Центральная зона обрезана: offset=(%d, %d), размер=%dx%d",
                        x_offset, y_offset, central_crop.shape[1], central_crop.shape[0]
                    )
                except Exception as e:
                    logger.warning("Ошибка обрезки центральной зоны: %s", e)
                    frame_for_ocr = processed
                    crop_offset = (0, 0)

        # === c) Детекция всплывающих диалогов (только если нет Florence) ===
        if self._florence_detector is None:
            try:
                popup_t0 = time.perf_counter()
                popup_result = detect_popup_dialog(processed)
                if popup_result.detected:
                    logger.info("Обнаружен всплывающий диалог — пропуск извлечения параметров")
                    # Возвращаем предыдущие значения если есть
                    if self._prev_params:
                        return self._prev_params.copy()
                popup_ms = (time.perf_counter() - popup_t0) * 1000
                ocr_log.log_stage("popup_detection", popup_ms)
            except Exception as e:
                logger.warning("Ошибка детекции диалога: %s", e)

        # === PRIMARY: Florence-first processing (Mode B) ===
        ocr_t0 = time.perf_counter()
        snapshot_params: dict[int, str] = {}

        # Пробуем Florence-2 первым (если доступен и не в Mode A)
        if self._florence_detector is not None and self._frames_since_tab_change >= FORCE_PROCESS_AFTER_TAB_CHANGE:
            try:
                florence_results = self._recognize_frame_florence(processed)
                if florence_results:
                    snapshot_params = florence_results
                    ocr_ms = (time.perf_counter() - ocr_t0) * 1000
                    logger.debug("Florence-first: %d параметров распознано", len(florence_results))
            except Exception as e:
                logger.warning("Ошибка Florence-first распознавания: %s", e)

        # FALLBACK 1: ROI-based (если Florence не дал результатов или недоступен)
        if not snapshot_params:
            roi_results = self._recognize_frame_roi(processed)
            if roi_results:
                snapshot_params = roi_results
                ocr_ms = (time.perf_counter() - ocr_t0) * 1000
                logger.debug("ROI-путь: %d параметров распознано", len(roi_results))

        # FALLBACK 2: Full-frame PaddleOCR (когда нет калибровки или ROI пуст)
        if not snapshot_params:
            snapshot_params = self._recognize_frame_with_logging(
                frame_for_ocr,
                zones,
                ocr_log,
                crop_offset=crop_offset,
                full_frame_shape=processed.shape[:2],
                full_frame=processed,
            )
            ocr_ms = (time.perf_counter() - ocr_t0) * 1000
            ocr_ms = (time.perf_counter() - ocr_t0) * 1000
            logger.debug("ROI-путь: %d параметров распознано", len(roi_results))
        else:
            # Fallback: полнокадровый OCR (когда нет калибровки или ROI пуст)
            snapshot_params = self._recognize_frame_with_logging(
                frame_for_ocr,
                zones,
                ocr_log,
                crop_offset=crop_offset,
                full_frame_shape=processed.shape[:2],
                full_frame=processed,
            )
            ocr_ms = (time.perf_counter() - ocr_t0) * 1000

        # Сохраняем для следующего кадра (для детекции дубликатов)
        self._prev_params = snapshot_params.copy()

        # Создаём снимок с метаданными
        param_metadata = self._get_param_metadata(list(snapshot_params.keys()))
        snapshot = create_snapshot(timestamp, snapshot_params, param_metadata)
        self._snapshots.append(snapshot)

        # Логируем результат кадра
        from app.core.ocr_models import RecognitionResult
        result = RecognitionResult(
            raw_fields={pid: str(val) for pid, val in snapshot_params.items()},
            confidence=0.85,  # Оценочное значение для pipeline режима
            source="pipeline_roi",
            pairs=[],
            frame_idx=idx,
            processing_ms=ocr_ms,
        )
        ocr_log.log_result(result, is_duplicate=False, fallback=None)

        # Пер-frame summary
        latency_ms = (time.perf_counter() - frame_start) * 1000
        logger.info(
            "Кадр %d/%d: %d параметров, %.0f мс (preprocess: %.1f, zones: %.1f, ocr: %.1f)",
            idx + 1, total_frames, len(snapshot_params), latency_ms,
            preprocess_ms, zone_ms, ocr_ms
        )

        # Инкрементируем счётчик кадров после обработки
        self._frames_since_tab_change += 1

        return snapshot_params

    def _run_parallel_pipeline(
        self,
        frames: list[tuple[np.ndarray, float]],
        video_type: VideoType,
        total_frames: int,
        ocr_log: OcrLogger,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        """Параллельная обработка кадров через конвейер стадий.

        Архитектура producer-consumer с 3 стадиями:
        1. Главный поток: извлечение, предобработка, дедупликация кадров
        2. OCR-поток: распознавание текста (PaddleOCR не thread-safe)
        3. Постобработка-потоки: извлечение значений, маппинг параметров

        Args:
            frames: Список (frame, timestamp_ms) для обработки.
            video_type: Тип видео.
            total_frames: Общее количество кадров.
            ocr_log: OCR логгер.
            progress_callback: Callback для отчёта о прогрессе.
        """
        pipeline_start = time.perf_counter()

        # Очереди между стадиями
        ocr_queue: queue.Queue = queue.Queue(maxsize=PARALLEL_QUEUE_SIZE)
        result_queue: queue.Queue = queue.Queue(maxsize=PARALLEL_QUEUE_SIZE * 2)

        # Событие остановки
        stop_event = threading.Event()

        # Статистика производительности
        stage_timings: dict[str, list[float]] = {
            "preprocess": [],
            "ocr": [],
            "postprocess": [],
        }
        stage_timings_lock = threading.Lock()

        # Результаты: frame_idx -> snapshot_params (для упорядочивания)
        results_lock = threading.Lock()
        results_ready: dict[int, dict[int, str]] = {}
        next_output_idx = [0]  # Используем список для изменяемости в замыкании
        output_lock = threading.Lock()

        # Импортируем metrics_collector
        from app.api.routes.pipeline import get_metrics_collector, _ws_broadcast_sync
        metrics_collector = get_metrics_collector()

        def _ocr_worker() -> None:
            """Рабочий поток OCR (единый для thread-safety PaddleOCR)."""
            while not stop_event.is_set():
                try:
                    item = ocr_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is _SENTINEL:
                    result_queue.put(_SENTINEL)
                    break

                idx, frame, timestamp, processed, zones, ts_str = item

                try:
                    ocr_t0 = time.perf_counter()
                    # OCR распознавание
                    snapshot_params = self._run_ocr_on_preprocessed(processed, zones, ocr_log)
                    ocr_ms = (time.perf_counter() - ocr_t0) * 1000

                    with stage_timings_lock:
                        stage_timings["ocr"].append(ocr_ms)

                    result_queue.put((idx, frame, timestamp, processed, snapshot_params, ts_str, ocr_ms))
                except Exception as e:
                    logger.error("OCR error frame %d: %s", idx, e)
                    result_queue.put((idx, frame, timestamp, processed, None, ts_str, 0.0))

        def _postprocess_worker() -> None:
            """Рабочий поток постобработки результатов OCR."""
            while not stop_event.is_set():
                try:
                    item = result_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is _SENTINEL:
                    break

                idx, frame, timestamp, processed, snapshot_params, ts_str, ocr_ms = item

                if snapshot_params is None:
                    continue

                try:
                    post_t0 = time.perf_counter()

                    # Сохраняем параметры для следующего кадра
                    self._prev_params = snapshot_params.copy()

                    # Создаём снимок с метаданными
                    param_metadata = self._get_param_metadata(list(snapshot_params.keys()))
                    snapshot = create_snapshot(timestamp, snapshot_params, param_metadata)
                    self._snapshots.append(snapshot)

                    # Логируем результат кадра
                    from app.core.ocr_models import RecognitionResult
                    result = RecognitionResult(
                        raw_fields={pid: str(val) for pid, val in snapshot_params.items()},
                        confidence=0.85,
                        source="parallel_pipeline",
                        pairs=[],
                        frame_idx=idx,
                        processing_ms=ocr_ms,
                    )
                    ocr_log.log_result(result, is_duplicate=False, fallback=None)

                    # Записываем метрики
                    post_ms = (time.perf_counter() - post_t0) * 1000
                    with stage_timings_lock:
                        stage_timings["postprocess"].append(post_ms)

                    frame_latency_ms = ocr_ms + post_ms
                    param_results = {pid: (val, None) for pid, val in snapshot_params.items()}
                    metrics_collector.record_frame(frame_latency_ms, param_results)

                    # Отправляем WebSocket-сообщения
                    for pid, val in snapshot_params.items():
                        meta = param_metadata.get(pid, ("", "", ""))
                        _ws_broadcast_sync({
                            "type": "ocr_result",
                            "video_id": "",
                            "param_id": pid,
                            "label": meta[0] or f"P{pid}",
                            "value": val,
                            "confidence": 0.85,
                            "source": "parallel_pipeline",
                            "processing_ms": frame_latency_ms,
                            "timestamp": ts_str,
                            "short_name": meta[0],
                            "full_name": meta[1],
                            "unit": meta[2],
                        })

                    # Периодические метрики
                    if idx % 10 == 0:
                        metrics_summary = metrics_collector.get_summary()
                        _ws_broadcast_sync({
                            "type": "metrics_update",
                            "video_id": "",
                            "metrics": metrics_summary,
                        })

                except Exception as e:
                    logger.error("Postprocess error frame %d: %s", idx, e)

        # Запускаем рабочие потоки
        ocr_thread = threading.Thread(target=_ocr_worker, daemon=True, name="OCR-Worker")
        post_threads = [
            threading.Thread(target=_postprocess_worker, daemon=True, name=f"PostProcess-{i}")
            for i in range(PARALLEL_POSTPROCESS_WORKERS)
        ]

        ocr_thread.start()
        for t in post_threads:
            t.start()

        # Главный поток: извлечение, предобработка, дедупликация
        prev_frame: np.ndarray | None = None
        processed_count = 0
        duplicate_count = 0

        for idx, (frame, timestamp) in enumerate(frames):
            self._skip_stats["total_frames"] += 1
            frame_start = time.perf_counter()
            ts_str = format_timestamp(timestamp / 1000.0) if isinstance(timestamp, (int, float)) else str(timestamp)

            ocr_log.start_frame(idx, ts_str)

            # Адаптивный пропуск
            self._frames_since_last_process += 1
            should_skip_adaptive = self._frames_since_last_process < self._adaptive_skip_interval

            if should_skip_adaptive and self._prev_params:
                self._skip_stats["skipped_by_adaptive"] += 1
                snapshot_params = self._prev_params.copy()
                self._duplicate_skip_count += 1

                param_metadata = self._get_param_metadata(list(snapshot_params.keys()))
                snapshot = create_snapshot(timestamp, snapshot_params, param_metadata)
                self._snapshots.append(snapshot)

                for pid, val in snapshot_params.items():
                    meta = param_metadata.get(pid, ("", "", ""))
                    _ws_broadcast_sync({
                        "type": "ocr_result",
                        "video_id": "",
                        "param_id": pid,
                        "label": meta[0] or f"P{pid}",
                        "value": val,
                        "confidence": 0.85,
                        "source": "adaptive_skip",
                        "processing_ms": 0.05,
                        "timestamp": ts_str,
                        "short_name": meta[0],
                        "full_name": meta[1],
                        "unit": meta[2],
                    })

                from app.core.ocr_models import RecognitionResult
                result = RecognitionResult(
                    raw_fields={pid: str(val) for pid, val in snapshot_params.items()},
                    confidence=0.85,
                    source="adaptive_skip",
                    pairs=[],
                    frame_idx=idx,
                    processing_ms=0.05,
                )
                ocr_log.log_result(result, is_duplicate=True, fallback=None)
                continue

            # Детекция вкладки SCADA
            tab_result = detect_active_scada_tab(frame, num_tabs=4)
            detected_tab = self._tab_tracker.update(tab_result)

            if has_tab_changed(self._get_tab_index_from_sheet(self._current_sheet_name), detected_tab):
                self._tab_change_count += 1
                prev_sheet = self._current_sheet_name
                new_sheet = self._get_sheet_name_from_tab(detected_tab)

                # Сохраняем текущую калибровку под старым именем вкладки
                # (или первичную калибровку если prev_sheet пуст)
                if prev_sheet and self._calibration:
                    self._tab_calibrations[prev_sheet] = self._calibration
                    logger.debug("Сохранена калибровка для вкладки '%s' (%d привязок)",
                                 prev_sheet, len(self._calibration.mappings))
                elif not prev_sheet and self._calibration:
                    # Первая детекция вкладки — сохраняем начальную калибровку
                    self._tab_calibrations[new_sheet] = self._calibration
                    logger.debug("Сохранена первичная калибровка под вкладкой '%s' (%d привязок)",
                                 new_sheet, len(self._calibration.mappings))

                # Обновляем имя текущей вкладки
                self._current_sheet_name = new_sheet
                self._frames_since_tab_change = 0  # Сброс счётчика кадров после смены вкладки

                logger.info(
                    "Смена вкладки SCADA: %s -> %s (вкладка %d)",
                    prev_sheet or "none", self._current_sheet_name, detected_tab
                )

                # Проверяем есть ли кэшированная калибровка для новой вкладки
                if new_sheet in self._tab_calibrations:
                    self._calibration = self._tab_calibrations[new_sheet]
                    logger.info("Восстановлена калибровка для вкладки '%s' (%d привязок)",
                                new_sheet, len(self._calibration.mappings))
                else:
                    # Новая вкладка — запускаем калибровку с Florence-2 grounding
                    logger.info("Новая вкладка '%s' — запуск калибровки", new_sheet)
                    try:
                        from app.core.florence_detector import FlorenceDetector

                        florence = self._get_florence_detector()
                        if florence is None:
                            florence = FlorenceDetector()
                            self._florence_detector = florence

                        new_mappings = calibrate_with_grounding(
                            frame, self._get_params_for_current_tab(), florence
                        )
                        if new_mappings:
                            new_profile = CalibrationProfile()
                            new_profile.mnemonic_name = new_sheet
                            for mapping in new_mappings:
                                new_profile.add_mapping(mapping)
                            self._calibration = new_profile
                            self._tab_calibrations[new_sheet] = new_profile
                            logger.info("Калибровка для '%s' завершена: %d привязок",
                                        new_sheet, len(new_mappings))
                        else:
                            # Fallback к существующему методу калибровки
                            self._calibrate(frame, video_type)
                            if self._calibration:
                                self._tab_calibrations[new_sheet] = self._calibration
                    except Exception as e:
                        logger.warning("Ошибка калибровки новой вкладки: %s — используем fallback", e)
                        self._calibrate(frame, video_type)
                        if self._calibration:
                            self._tab_calibrations[new_sheet] = self._calibration

                # Сбрасываем историю confidence scorer при смене вкладки
                if hasattr(self._confidence_scorer, 'reset_history'):
                    self._confidence_scorer.reset_history()

                self._prev_params = {}
                self._reset_adaptive_skip("tab_change")

            # Дедупликация
            is_duplicate, is_scene_change = self._is_duplicate_frame(frame)

            if is_scene_change:
                logger.info("Смена сцены на кадре %d — принудительная обработка", idx + 1)
                self._reset_adaptive_skip("scene_change")

            if is_duplicate and self._prev_params and not is_scene_change:
                duplicate_count += 1
                self._duplicate_skip_count += 1
                self._skip_stats["skipped_by_duplicate"] += 1
                self._consecutive_duplicates += 1

                if self._consecutive_duplicates >= ADAPTIVE_SKIP_DUPLICATE_THRESHOLD:
                    self._increase_adaptive_skip()
                    self._consecutive_duplicates = 0

                snapshot_params = self._prev_params.copy()
                param_metadata = self._get_param_metadata(list(snapshot_params.keys()))
                snapshot = create_snapshot(timestamp, snapshot_params, param_metadata)
                self._snapshots.append(snapshot)

                for pid, val in snapshot_params.items():
                    meta = param_metadata.get(pid, ("", "", ""))
                    _ws_broadcast_sync({
                        "type": "ocr_result",
                        "video_id": "",
                        "param_id": pid,
                        "label": meta[0] or f"P{pid}",
                        "value": val,
                        "confidence": 0.85,
                        "source": "pipeline_roi_duplicate",
                        "processing_ms": 0.1,
                        "timestamp": ts_str,
                        "short_name": meta[0],
                        "full_name": meta[1],
                        "unit": meta[2],
                    })

                from app.core.ocr_models import RecognitionResult
                result = RecognitionResult(
                    raw_fields={pid: str(val) for pid, val in snapshot_params.items()},
                    confidence=0.85,
                    source="pipeline_roi_duplicate",
                    pairs=[],
                    frame_idx=idx,
                    processing_ms=0.1,
                )
                ocr_log.log_result(result, is_duplicate=True, fallback=None)
                prev_frame = frame
                continue

            # Сброс счётчика дубликатов при уникальном кадре
            if not is_duplicate:
                if self._consecutive_duplicates > 0:
                    logger.debug(
                        "Сброс счётчика дубликатов (%d) — обработан уникальный кадр",
                        self._consecutive_duplicates
                    )
                self._consecutive_duplicates = 0
                if self._adaptive_skip_interval > ADAPTIVE_SKIP_INITIAL:
                    self._reset_adaptive_skip("unique_frame_detected")

            # Предобработка
            preprocess_t0 = time.perf_counter()
            processed, _scale = preprocess_frame(frame, video_type, prev_frame, self._screen_corners)
            preprocess_ms = (time.perf_counter() - preprocess_t0) * 1000
            ocr_log.log_stage("preprocess", preprocess_ms)
            with stage_timings_lock:
                stage_timings["preprocess"].append(preprocess_ms)

            # Сегментация зон
            zone_t0 = time.perf_counter()
            zones = segment_zones(processed)
            zone_ms = (time.perf_counter() - zone_t0) * 1000
            ocr_log.log_stage("zone_segmentation", zone_ms)

            # Отправляем в OCR-очередь
            self._skip_stats["processed_frames"] += 1
            self._frames_since_last_process = 0
            processed_count += 1

            ocr_queue.put((idx, frame, timestamp, processed, zones, ts_str))
            prev_frame = frame

            # Прогресс
            if idx % 10 == 0:
                progress = (idx + 1) / total_frames * 100
                logger.info("Обработано %d/%d кадров (%.1f%%)", idx + 1, total_frames, progress)
            if progress_callback:
                progress_callback(idx + 1, total_frames, "ocr")

        # Сигнал завершения
        ocr_queue.put(_SENTINEL)

        # Ожидаем завершение OCR-потока
        ocr_thread.join(timeout=300.0)

        # Сигнал завершения для постобработки
        for _ in post_threads:
            result_queue.put(_SENTINEL)

        # Ожидаем завершение постобработки
        for t in post_threads:
            t.join(timeout=60.0)

        pipeline_elapsed = time.perf_counter() - pipeline_start

        # Логируем статистику
        logger.info(
            "Pipeline: %d total frames, %d processed, %d duplicates, %.1fs total, "
            "%.2fs/frame avg",
            total_frames, processed_count, duplicate_count, pipeline_elapsed,
            pipeline_elapsed / max(1, processed_count)
        )

        with stage_timings_lock:
            if stage_timings["preprocess"]:
                avg_preprocess = sum(stage_timings["preprocess"]) / len(stage_timings["preprocess"])
                logger.info("Stage timings: preprocess=%.1fms avg", avg_preprocess)
            if stage_timings["ocr"]:
                avg_ocr = sum(stage_timings["ocr"]) / len(stage_timings["ocr"])
                logger.info("Stage timings: ocr=%.1fms avg", avg_ocr)
            if stage_timings["postprocess"]:
                avg_post = sum(stage_timings["postprocess"]) / len(stage_timings["postprocess"])
                logger.info("Stage timings: postprocess=%.1fms avg", avg_post)

    def _run_ocr_on_preprocessed(
        self,
        processed: np.ndarray,
        zones: list,
        ocr_log: OcrLogger,
    ) -> dict[int, str]:
        """Выполняет OCR на предобработанном кадре.

        Вызывается из OCR-потока параллельного конвейера.
        Florence-first стратегия: сначала Florence-2, затем ROI, затем полнокадровый OCR.

        Args:
            processed: Предобработанный кадр.
            zones: Зоны мнемосхемы.
            ocr_log: OCR логгер.

        Returns:
            Словарь param_id -> отформатированное значение.
        """
        # PRIMARY: Florence-first processing (Mode B, если не в фазе Mode A)
        if self._florence_detector is not None and self._frames_since_tab_change >= FORCE_PROCESS_AFTER_TAB_CHANGE:
            try:
                florence_t0 = time.perf_counter()
                logger.debug("Florence-путь: обработка кадра через OCR_WITH_REGION + fuzzy match")
                florence_results = self._recognize_frame_florence(processed)
                florence_ms = (time.perf_counter() - florence_t0) * 1000
                ocr_log.log_stage("florence_ocr_parallel", florence_ms)
                if florence_results:
                    logger.debug("Florence-first (parallel): %d параметров", len(florence_results))
                    return florence_results
                else:
                    logger.debug("Florence не вернул результатов — fallback к ROI/PaddleOCR")
            except Exception as e:
                logger.warning("Ошибка Florence-first в parallel pipeline: %s — fallback к ROI", e)

        # FALLBACK 1: ROI-быстрый путь
        roi_results = self._recognize_frame_roi(processed)
        if roi_results:
            logger.debug("ROI-путь (parallel): %d параметров", len(roi_results))
            return roi_results

        # FALLBACK 2: Полный OCR pipeline
        logger.debug("Fallback: полный OCR pipeline через _recognize_frame_with_logging")
        return self._recognize_frame_with_logging(processed, zones, ocr_log)

    def _get_param_metadata(self, param_ids: list[int]) -> dict[int, tuple[str, str, str]]:
        """Возвращает метаданные параметров по их ID.

        Args:
            param_ids: Список ID параметров.

        Returns:
            Словарь param_id -> (short_name, full_name, unit).
        """
        metadata: dict[int, tuple[str, str, str]] = {}
        if not self._calibration:
            return metadata
        for mapping in self._calibration.mappings:
            if mapping.param_id in param_ids:
                metadata[mapping.param_id] = (
                    mapping.short_name,
                    mapping.full_name,
                    mapping.unit,
                )
        return metadata

    def _is_duplicate_frame(self, frame: np.ndarray) -> tuple[bool, bool]:
        """Проверяет, является ли кадр дубликатом предыдущего (3-tier hybrid detector).

        Tier 1: Quick pixel difference — быстрый ранний фильтр для почти идентичных кадров.
        Tier 2: ROI-based MSE — основная детекция изменений в зонах параметров из калибровки.
        Tier 3: Scene change detection — проверка верхней части на смену вкладки/сцены.

        Args:
            frame: Текущий кадр BGR.

        Returns:
            Кортеж (is_duplicate, is_scene_change):
            - is_duplicate: True если кадр считается дубликатом
            - is_scene_change: True если обнаружена смена сцены
        """
        # === Forced Processing Checks ===
        # Проверяем необходимость принудительной обработки независимо от дубликатов
        current_time = time.time()
        time_since_last_process = current_time - self._last_processed_timestamp

        if self._last_processed_timestamp > 0 and time_since_last_process > MIN_PROCESS_INTERVAL_SEC:
            logger.info(
                "Принудительная обработка: прошло %.1f сек с последней обработки (порог: %.1f)",
                time_since_last_process, MIN_PROCESS_INTERVAL_SEC
            )
            return False, False

        if self._frames_since_tab_change < FORCE_PROCESS_AFTER_TAB_CHANGE:
            logger.info(
                "Принудительная обработка: кадр %d после смены вкладки (порог: %d)",
                self._frames_since_tab_change, FORCE_PROCESS_AFTER_TAB_CHANGE
            )
            return False, False

        if self._consecutive_duplicate_count >= MAX_CONSECUTIVE_DUPLICATES:
            logger.info(
                "Принудительная обработка: достигнут лимит последовательных дубликатов (%d)",
                self._consecutive_duplicate_count
            )
            return False, False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            return False, False

        if gray.shape != self._prev_frame_gray.shape:
            self._prev_frame_gray = gray
            return False, False

        # === Tier 3: Scene change detection (выполняем первым для определения размеров) ===
        h, w = gray.shape
        top_h = int(h * SCENE_CHANGE_TOP_REGION_RATIO)
        top_curr = gray[:top_h, :]
        top_prev = self._prev_frame_gray[:top_h, :]

        # Уменьшаем для скорости
        top_small_h, top_small_w = 16, 64
        top_small_curr = cv2.resize(top_curr, (top_small_w, top_small_h))
        top_small_prev = cv2.resize(top_prev, (top_small_w, top_small_h))
        mse_top = float(np.mean((top_small_curr.astype(np.float32) - top_small_prev.astype(np.float32)) ** 2))

        # Определяем смену сцены по верхней части
        is_scene_change = mse_top > SCENE_CHANGE_MSE_THRESHOLD

        if is_scene_change:
            self._prev_frame_gray = gray
            logger.info(
                "Обнаружена смена сцены/вкладки (MSE top=%.1f > %.1f), кадр будет обработан",
                mse_top, SCENE_CHANGE_MSE_THRESHOLD
            )
            return False, True

        # === Tier 1: Quick pixel difference (fast early filter) ===
        diff = cv2.absdiff(gray, self._prev_frame_gray)
        _, binary = cv2.threshold(diff, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        changed_pixels = cv2.countNonZero(binary)

        if changed_pixels < PIXEL_DIFF_MIN:
            # Почти идентичные кадры — точно дубликат
            self._prev_frame_gray = gray
            logger.debug(
                "Кадр классифицирован как ДУБЛИКАТ (Tier 1: changed_pixels=%d < %d)",
                changed_pixels, PIXEL_DIFF_MIN
            )
            return True, False

        # === Tier 2: ROI-based MSE (primary detector using calibration ROIs) ===
        max_roi_mse = 0.0
        changed_roi_count = 0
        total_rois = 0

        if self._calibration and hasattr(self._calibration, 'mappings') and self._calibration.mappings:
            for mapping in self._calibration.mappings:
                roi = mapping.roi_bbox  # BoundingBox с нормализованными координатами [0,1]
                total_rois += 1

                # Конвертируем нормализованные координаты в пиксельные
                x1 = int(roi.x1 * w)
                y1 = int(roi.y1 * h)
                x2 = int(roi.x2 * w)
                y2 = int(roi.y2 * h)

                # Проверяем валидность ROI
                if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    continue

                # Извлекаем ROI из обоих кадров
                roi_curr = gray[y1:y2, x1:x2]
                roi_prev = self._prev_frame_gray[y1:y2, x1:x2]

                if roi_curr.size == 0 or roi_prev.size == 0:
                    continue

                # Вычисляем MSE для ROI
                roi_mse = float(np.mean((roi_curr.astype(np.float32) - roi_prev.astype(np.float32)) ** 2))
                max_roi_mse = max(max_roi_mse, roi_mse)

                if roi_mse > ROI_MSE_THRESHOLD:
                    changed_roi_count += 1
                    # Любое значимое изменение в ROI → уникальный кадр
                    self._prev_frame_gray = gray
                    logger.debug(
                        "Кадр классифицирован как УНИКАЛЬНЫЙ "
                        "(Tier 2: ROI MSE=%.1f > %.1f, param_id=%d, changed_rois=%d/%d)",
                        roi_mse, ROI_MSE_THRESHOLD, mapping.param_id, changed_roi_count, total_rois
                    )
                    return False, False

        # Если есть калибровка и ROI не изменились → дубликат
        if total_rois > 0:
            self._prev_frame_gray = gray
            logger.info(
                "Кадр классифицирован как ДУБЛИКАТ "
                "(Tier 2: max_roi_mse=%.1f <= %.1f, changed_rois=%d/%d)",
                max_roi_mse, ROI_MSE_THRESHOLD, changed_roi_count, total_rois
            )
            return True, False

        # === Fallback: Full-frame MSE when no calibration ===
        small_h, small_w = 64, 64
        small_curr = cv2.resize(gray, (small_w, small_h))
        small_prev = cv2.resize(self._prev_frame_gray, (small_w, small_h))

        mse_full = float(np.mean((small_curr.astype(np.float32) - small_prev.astype(np.float32)) ** 2))

        self._prev_frame_gray = gray

        is_duplicate = mse_full < DUPLICATE_MSE_THRESHOLD

        if is_duplicate:
            logger.debug(
                "Кадр классифицирован как ДУБЛИКАТ (Fallback: MSE full=%.1f < %.1f)",
                mse_full, DUPLICATE_MSE_THRESHOLD
            )
        else:
            logger.debug(
                "Кадр классифицирован как УНИКАЛЬНЫЙ (Fallback: MSE full=%.1f >= %.1f)",
                mse_full, DUPLICATE_MSE_THRESHOLD
            )

        return is_duplicate, False

    def _get_sheet_name_from_tab(self, tab_index: int) -> str:
        """Преобразует индекс вкладки в имя листа Excel.

        Args:
            tab_index: Индекс вкладки (0-3).

        Returns:
            Имя листа ("1_ai", "2_ai", "3_ai", "4_ai") или пустая строка.
        """
        sheet_names = ["1_ai", "2_ai", "3_ai", "4_ai"]
        if 0 <= tab_index < len(sheet_names):
            return sheet_names[tab_index]
        return ""

    def _get_tab_index_from_sheet(self, sheet_name: str) -> int:
        """Преобразует имя листа в индекс вкладки.

        Args:
            sheet_name: Имя листа ("1_ai", "2_ai", etc.).

        Returns:
            Индекс вкладки (0-3) или -1.
        """
        sheet_names = ["1_ai", "2_ai", "3_ai", "4_ai"]
        try:
            return sheet_names.index(sheet_name)
        except ValueError:
            return -1

    def _get_params_for_current_tab(self) -> list[dict]:
        """Возвращает параметры для текущей активной вкладки и типа ГПА.

        Фильтрует параметры по sheet_name (вкладка) и gpa_type (тип ГПА).
        GPA-11 и GPA-21 имеют разные наборы параметров.

        Returns:
            Список параметров, отфильтрованных по sheet_name и gpa_type.
        """
        if not self._param_table:
            return []

        # Базовая фильтрация по вкладке
        if self._current_sheet_name:
            params = [
                p for p in self._param_table
                if p.get("sheet_name", "") == self._current_sheet_name
            ]
        else:
            params = self._param_table

        # Дополнительная фильтрация по типу ГПА если известен
        if self._current_gpa_type:
            # GPA-11 и GPA-21 имеют разные параметры
            # Фильтруем по gpa_type если поле есть в таблице
            gpa_filtered = [
                p for p in params
                if p.get("gpa_type") is None or p.get("gpa_type") == self._current_gpa_type
            ]
            if gpa_filtered:  # Используем фильтрованный список только если не пустой
                params = gpa_filtered

        return params

    def _calibrate(self, first_frame: np.ndarray, video_type: VideoType) -> None:
        """Калибровка на первом кадре через Florence-2 (primary) или PaddleOCR (fallback).

        PRIMARY: Florence-based калибровка (7-фазная) — точное позиционирование
        меток параметров через CAPTION_TO_PHRASE_GROUNDING.
        FALLBACK: Старый PaddleOCR подход — spatial clustering + fuzzy matching.

        Args:
            first_frame: Первый кадр.
            video_type: Тип видео.
        """
        # Предобработка первого кадра
        if self._progress_callback:
            self._progress_callback(0, 0, "preprocessing")
        processed, _scale = preprocess_frame(first_frame, video_type)

        # PRIMARY: Florence-based калибровка (7-фазная)
        try:
            self._get_florence_detector()
            if self._florence_detector is not None and self._param_table:
                if self._progress_callback:
                    self._progress_callback(0, 0, "florence_calibration")
                mappings = calibrate_with_grounding(
                    processed, self._param_table, self._florence_detector
                )
                if mappings:
                    self._calibration = CalibrationProfile()
                    for mapping in mappings:
                        self._calibration.add_mapping(mapping)
                    # Сохраняем scene info для отслеживания вкладок
                    try:
                        scene = self._florence_detector.describe_scene(processed)
                        self._current_tab = scene.get("active_tab")
                        self._current_gpa_type = scene.get("gpa_type")
                    except Exception:
                        pass
                    logger.info(
                        "Florence калибровка: %d привязок к параметрам",
                        len(mappings),
                    )
                    return
                else:
                    logger.warning(
                        "Florence калибровка не нашла привязок — fallback к PaddleOCR"
                    )
        except Exception as e:
            logger.warning("Florence калибровка ошибка: %s — fallback к PaddleOCR", e)

        # FALLBACK: Старый PaddleOCR подход
        from app.core.ocr_engine import ocr_full_frame_enhanced

        # Сегментация зон
        if self._progress_callback:
            self._progress_callback(0, 0, "zone_segmentation")
        zones = segment_zones(processed)

        # Детекция экрана для handheld
        if video_type != VideoType.DIRECT:
            if self._progress_callback:
                self._progress_callback(0, 0, "screen_detection")
            from app.core.screen_detector import detect_screen

            corners = detect_screen(first_frame)
            if corners is not None:
                self._screen_corners = corners

        # OCR для калибровки: используем быстрый путь (только PaddleOCR).
        # Florence-2 слишком дорога для калибровки (3-5с на кадр) и запустится
        # автоматически во время обработки кадров по расписанию (florence_interval_sec).
        # Для калибровки достаточно PaddleOCR — нужно лишь определить расположение
        # текстов на мнемосхеме и сопоставить их с таблицей параметров.
        if self._progress_callback:
            self._progress_callback(0, 0, "ocr_paddle")
        ocr_results = ocr_full_frame(processed)

        # Извлекаем пары через spatial clusterer (быстро, без Florence)
        if self._progress_callback:
            self._progress_callback(0, 0, "spatial_clustering")
        central_zone = next(
            (z for z in zones if z.zone == ZoneType.CENTRAL_SCHEMA), None
        )
        pairs = cluster_label_value_pairs(ocr_results, processed.shape[:2])

        # Парсинг правой панели
        right_zone = next(
            (z for z in zones if z.zone == ZoneType.RIGHT_PANEL), None
        )
        right_pairs: list = []
        if right_zone:
            right_pairs = parse_right_panel(ocr_results, right_zone.bbox, processed.shape[:2])

        # Объединяем пары
        all_pairs = pairs + right_pairs

        # Сопоставление с таблицей параметров
        if self._progress_callback:
            self._progress_callback(0, 0, "parameter_mapping")
        mappings = match_labels_to_params(all_pairs, self._param_table)

        # Создаём профиль калибровки
        self._calibration = CalibrationProfile()
        self._calibration.zones = zones
        for mapping in mappings:
            self._calibration.add_mapping(mapping)

        logger.info(
            "Калибровка: %d пар label-value (spatial: %d, right: %d), %d привязок к параметрам",
            len(all_pairs), len(pairs), len(right_pairs),
            len(mappings),
        )

    def _get_florence_detector(self):
        """Ленивая загрузка Florence-2 детектора (singleton).

        Использует глобальный singleton из florence_detector модуля
        для предотвращения создания множественных экземпляров.

        Returns:
            FlorenceDetector instance или None если недоступен.
        """
        if self._florence_detector is None:
            try:
                from app.core.florence_detector import _load_florence, FlorenceDetector

                if FlorenceDetector.is_available():
                    # Используем singleton _load_florence вместо создания нового экземпляра
                    _load_florence()  # Инициализирует глобальный singleton
                    self._florence_detector = FlorenceDetector()
                    logger.info("Florence-2 детектор инициализирован для верификации (singleton)")
                else:
                    logger.debug("Florence-2 недоступен — верификация отключена")
            except Exception as e:
                logger.warning("Ошибка загрузки Florence-2: %s", e)
        return self._florence_detector

    def _parse_florence_numeric_values(self, text: str) -> list[tuple[str, float]]:
        """Парсит числовые значения из текста Florence-2.

        Args:
            text: Распознанный текст от Florence-2.

        Returns:
            Список кортежей (число_как_строка, значение).
        """
        values = []
        # Ищем числа: целые и с плавающей точкой, включая отрицательные
        pattern = r'[-+]?\d+\.?\d*'
        for match in re.finditer(pattern, text):
            num_str = match.group()
            try:
                num_val = float(num_str)
                values.append((num_str, num_val))
            except ValueError:
                continue
        return values

    def _check_florence_discrepancy(
        self,
        paddle_value: str,
        florence_text: str,
        roi_bbox: tuple,
    ) -> tuple[bool, str, float]:
        """Проверяет расхождение между PaddleOCR и Florence-2.

        Args:
            paddle_value: Значение от PaddleOCR.
            florence_text: Полный текст от Florence-2 OCR.
            roi_bbox: ROI bounding box (x, y, w, h).

        Returns:
            Кортеж (есть_расхождение, предпочтительное_значение, confidence_multiplier).
        """
        # Парсим числовые значения из обоих источников
        paddle_nums = self._parse_florence_numeric_values(paddle_value)
        florence_nums = self._parse_florence_numeric_values(florence_text)

        if not paddle_nums or not florence_nums:
            return False, paddle_value, 1.0

        # Берём первое числовое значение из Paddle
        paddle_num = paddle_nums[0][1]

        # Ищем ближайшее значение во Florence тексте
        best_match = None
        best_diff = float('inf')

        for fl_str, fl_val in florence_nums:
            diff = abs(fl_val - paddle_num)
            if diff < best_diff:
                best_diff = diff
                best_match = fl_str

        if best_match is None:
            return False, paddle_value, 1.0

        # Вычисляем относительное расхождение
        if paddle_num != 0:
            relative_diff = best_diff / abs(paddle_num)
        else:
            relative_diff = best_diff

        # Порог 5% для расхождения
        threshold = 0.05

        if relative_diff > threshold:
            # Florence даёт другое значение — логируем расхождение
            logger.info(
                "Florence верификация: расхождение %.1f%% (Paddle: %s, Florence: %s)",
                relative_diff * 100, paddle_value, best_match
            )
            # Предпочитаем Florence (он точнее на сложных кадрах)
            return True, best_match, 0.8

        # Значения совпадают — повышаем confidence
        return False, paddle_value, 1.1

    def _find_value_near_label(
        self,
        label_bbox: BoundingBox,
        all_ocr_results: list[OCRTextResult],
        frame_shape: tuple[int, int],
        max_horizontal_distance: float = 0.15,  # Normalized (15% of frame width)
        max_vertical_distance: float = 0.05,    # Normalized (5% of frame height)
    ) -> tuple[float | None, float, str]:
        """Ищет числовое значение рядом с меткой параметра.

        SCADA-экраны имеют LABELS и VALUES как отдельные текстовые элементы.
        Значение обычно справа или под меткой.

        Приоритет поиска:
        1. Текстовые боксы СПРАВА от метки (на том же уровне по Y, ±30px)
        2. Текстовые боксы ПОД меткой (на том же уровне по X)
        3. Любой бокс в пределах max_distance, содержащий число

        Args:
            label_bbox: Bbox метки параметра (нормализованные координаты).
            all_ocr_results: Все результаты OCR на кадре.
            frame_shape: Размер кадра (height, width).
            max_horizontal_distance: Макс. расстояние по горизонтали (нормализованное).
            max_vertical_distance: Макс. расстояние по вертикали (нормализованное).

        Returns:
            Кортеж (найденное_число, confidence, позиция_относительно_метки).
            Позиция: 'right', 'below', или 'fallback'.
        """
        h, w = frame_shape[:2]

        # Центр и правый край метки
        label_cx = (label_bbox.x1 + label_bbox.x2) / 2
        label_cy = (label_bbox.y1 + label_bbox.y2) / 2
        label_right = label_bbox.x2  # Правый край метки
        label_bottom = label_bbox.y2  # Нижний край метки

        # Кандидаты: (число, confidence, расстояние, позиция, bbox)
        candidates_right: list[tuple[float, float, float, str, BoundingBox]] = []
        candidates_below: list[tuple[float, float, float, str, BoundingBox]] = []
        candidates_fallback: list[tuple[float, float, float, str, BoundingBox]] = []

        for ocr_result in all_ocr_results:
            # Пропускаем саму метку (перекрывающиеся боксы)
            text_bbox = ocr_result.bbox
            # Проверяем перекрытие с меткой
            overlap_x = max(0, min(label_bbox.x2, text_bbox.x2) - max(label_bbox.x1, text_bbox.x1))
            overlap_y = max(0, min(label_bbox.y2, text_bbox.y2) - max(label_bbox.y1, text_bbox.y1))
            overlap_area = overlap_x * overlap_y
            text_area = (text_bbox.x2 - text_bbox.x1) * (text_bbox.y2 - text_bbox.y1)
            if text_area > 0 and overlap_area / text_area > 0.5:
                # Бокс в основном перекрывается с меткой — пропускаем
                continue

            # Проверяем, содержит ли текст число
            number = ValueProcessor._extract_best_number(ocr_result.text)
            if number is None:
                continue

            # Центр текстового бокса
            text_cx = (text_bbox.x1 + text_bbox.x2) / 2
            text_cy = (text_bbox.y1 + text_bbox.y2) / 2

            # Расстояние от правого края метки
            dx = text_cx - label_right
            dy = text_cy - label_cy

            # Приоритет 1: Справа от метки (горизонтальное расположение)
            # Значение должно быть правее метки, на том же уровне по Y
            if dx > 0 and abs(dy) < max_vertical_distance:
                if dx < max_horizontal_distance:
                    distance = dx + abs(dy) * 0.1  # Небольшой штраф за вертикальное смещение
                    candidates_right.append((number, ocr_result.confidence, distance, 'right', text_bbox))

            # Приоритет 2: Под меткой (вертикальное расположение)
            # Значение должно быть ниже метки, на том же уровне по X
            vertical_offset = text_cy - label_bottom
            if vertical_offset > 0 and vertical_offset < max_vertical_distance * 2:
                if abs(text_cx - label_cx) < max_horizontal_distance * 0.5:
                    distance = vertical_offset + abs(text_cx - label_cx) * 0.1
                    candidates_below.append((number, ocr_result.confidence, distance, 'below', text_bbox))

            # Приоритет 3: Fallback — любой бокс рядом с меткой
            euclidean_dist = (dx ** 2 + dy ** 2) ** 0.5
            if euclidean_dist < max_horizontal_distance:
                candidates_fallback.append((number, ocr_result.confidence, euclidean_dist, 'fallback', text_bbox))

        # Выбираем лучший кандидат (приоритет: right > below > fallback)
        if candidates_right:
            candidates_right.sort(key=lambda c: c[2])  # Сортируем по расстоянию
            best = candidates_right[0]
            return best[0], best[1], best[3]

        if candidates_below:
            candidates_below.sort(key=lambda c: c[2])
            best = candidates_below[0]
            return best[0], best[1], best[3]

        if candidates_fallback:
            candidates_fallback.sort(key=lambda c: c[2])
            best = candidates_fallback[0]
            return best[0], best[1], best[3]

        return None, 0.0, 'none'

    def _filter_ocr_by_roi(
        self,
        all_ocr_results: list[OCRTextResult],
        roi: BoundingBox,
        frame_shape: tuple[int, int],
    ) -> tuple[str | None, float]:
        """Фильтрует результаты OCR по ROI без дополнительного OCR вызова.

        Использует уже полученные результаты full-frame OCR и фильтрует их по ROI.
        Это заменяет вызов ocr_roi(), который запускал бы новый OCR на каждом параметре.

        Args:
            all_ocr_results: Все результаты OCR на кадре.
            roi: Область интереса в нормализованных координатах.
            frame_shape: Размер кадра (height, width).

        Returns:
            Кортеж (найденный_текст, confidence) или (None, 0.0).
        """
        h, w = frame_shape[:2]

        # Конвертируем нормализованные координаты ROI в пиксельные
        roi_x1 = roi.x1 * w
        roi_y1 = roi.y1 * h
        roi_x2 = roi.x2 * w
        roi_y2 = roi.y2 * h
        roi_cx = (roi_x1 + roi_x2) / 2
        roi_cy = (roi_y1 + roi_y2) / 2

        # Ищем OCR результаты, которые попадают в ROI
        candidates: list[tuple[str, float, float]] = []  # (text, confidence, distance_to_center)

        for ocr_result in all_ocr_results:
            # Конвертируем нормализованные координаты OCR в пиксельные
            text_x1 = ocr_result.bbox.x1 * w
            text_y1 = ocr_result.bbox.y1 * h
            text_x2 = ocr_result.bbox.x2 * w
            text_y2 = ocr_result.bbox.y2 * h
            text_cx = (text_x1 + text_x2) / 2
            text_cy = (text_y1 + text_y2) / 2

            # Проверяем, попадает ли центр текста в ROI
            if roi_x1 <= text_cx <= roi_x2 and roi_y1 <= text_cy <= roi_y2:
                distance = ((text_cx - roi_cx) ** 2 + (text_cy - roi_cy) ** 2) ** 0.5
                candidates.append((ocr_result.text, ocr_result.confidence, distance))

        if not candidates:
            return None, 0.0

        # Сортируем по confidence (выше лучше), затем по расстоянию (меньше лучше)
        candidates.sort(key=lambda c: (-c[1], c[2]))
        best = candidates[0]
        return best[0], best[1]

    def _recognize_frame_roi(self, frame: np.ndarray) -> dict[int, str]:
        """Распознаёт значения параметров по ROI из калибровки.

        Быстрый путь обработки: вместо полнокадрового OCR использует
        заранее определённые ROI (value_bbox) из калибровки для каждого параметра.
        Это значительно быстрее и точнее, так как OCR применяется только
        к маленьким областям с известными значениями.

        Args:
            frame: Предобработанный кадр BGR.

        Returns:
            Словарь param_id -> распознанное значение.
            Пустой словарь если нет калибровки или ROI.
        """
        # Проверяем наличие калибровки с маппингами
        if not self._calibration or not self._calibration.mappings:
            return {}

        h, w = frame.shape[:2]
        roi_list: list[tuple[int, tuple[int, int, int, int], str, str]] = []

        # Собираем ROI из калибровки
        for mapping in self._calibration.mappings:
            value_bbox = mapping.value_bbox
            if value_bbox is None:
                continue

            # Конвертируем нормализованные координаты в пиксели
            x1 = max(0, int(value_bbox.x1 * w))
            y1 = max(0, int(value_bbox.y1 * h))
            x2 = min(w, int(value_bbox.x2 * w))
            y2 = min(h, int(value_bbox.y2 * h))

            roi_w = x2 - x1
            roi_h = y2 - y1

            # Пропускаем слишком маленькие ROI
            if roi_w < 5 or roi_h < 5:
                continue

            roi_list.append(
                (mapping.param_id, (x1, y1, roi_w, roi_h), mapping.short_name, mapping.unit)
            )

        if not roi_list:
            return {}

        # Выполняем batch OCR по ROI
        rois = [roi for _, roi, _, _ in roi_list]
        ocr_results = ocr_roi_batch(frame, rois, min_confidence=0.5)

        # Обрабатываем результаты
        results: dict[int, str] = {}
        for (param_id, _, short_name, unit), (text, confidence) in zip(roi_list, ocr_results):
            if not text:
                continue

            # Валидируем значение по типу параметра
            cleaned_text, is_valid = validate_parameter_value(text, short_name, unit)

            if not is_valid:
                logger.debug(
                    "ROI OCR: параметр %d — значение '%s' не прошло валидацию (тип=%s)",
                    param_id, text, short_name
                )
                continue

            # Применяем temporal smoothing если доступен confidence_scorer
            if hasattr(self, '_confidence_scorer') and self._confidence_scorer:
                try:
                    smoothed_value, smoothed_conf = self._confidence_scorer.smooth_value(
                        param_id=param_id,
                        current_value=cleaned_text,
                        current_confidence=confidence,
                    )
                    if smoothed_value is not None:
                        cleaned_text = smoothed_value
                        confidence = smoothed_conf
                except Exception as e:
                    logger.debug("Ошибка smoothing для param %d: %s", param_id, e)

            results[param_id] = cleaned_text
            logger.debug(
                "ROI OCR: параметр %d = '%s' (conf=%.2f, тип=%s)",
                param_id, cleaned_text, confidence, short_name
            )

        return results

    def _recognize_frame_florence(self, frame: np.ndarray) -> dict[int, str]:
        """Распознаёт значения параметров на кадре через Florence-2 OCR (Mode B).

        Альтернативный путь обработки: использует Florence-2 OCR_WITH_REGION
        для полного распознавания кадра с координатами и confidence.
        Выполняет fuzzy matching меток против таблицы параметров и ищет
        числовые значения рядом с распознанными метками.

        Стратегия:
        1. Florence-2 OCR_WITH_REGION → список (text, bbox, confidence)
        2. Fuzzy matching меток через _compute_match_score против param_table
        3. Поиск числовых значений рядом с метками (справа или снизу)
        4. Валидация и temporal smoothing через confidence_scorer

        Args:
            frame: Предобработанный кадр BGR.

        Returns:
            Словарь param_id -> распознанное значение.
            Пустой словарь при ошибке или отсутствии данных.
        """
        # Проверяем наличие Florence-детектора и таблицы параметров
        if self._florence_detector is None:
            logger.debug("Florence-2 детектор не инициализирован — пропускаем")
            return {}

        if not self._param_table:
            logger.debug("Таблица параметров пуста — нечего сопоставлять")
            return {}

        try:
            # === Шаг 1: OCR_WITH_REGION с confidence ===
            ocr_results = self._florence_detector.ocr_all_text(frame)
            if not ocr_results:
                logger.debug("Florence-2 не нашёл текст на кадре")
                return {}

            # Предфильтрация: пропускаем single char и низкий confidence
            filtered_results: list[tuple[str, "BBox", float]] = []
            for text, bbox, conf in ocr_results:
                if len(text.strip()) <= 1:
                    continue
                if conf < 0.3:
                    continue
                filtered_results.append((text, bbox, conf))

            if not filtered_results:
                logger.debug("Все OCR-результаты отфильтрованы (single char или low conf)")
                return {}

            # Логируем первые OCR-результаты для отладки
            logger.debug("Florence OCR: %d результатов после фильтрации", len(filtered_results))
            for text, bbox, conf in filtered_results[:5]:
                logger.debug("  OCR текст: '%s' (conf=%.2f, bbox=%s)", text, conf, bbox)

            # Regex для числовых значений (пропускаем при matching — это значения, не метки)
            numeric_only = re.compile(r'^[+-]?\d+[.,]?\d*$')

            # === Шаг 2: Fuzzy matching меток против param_table ===
            # Структура: param_id -> (label_text, label_bbox, score, conf)
            label_matches: dict[int, tuple[str, "BBox", float, float]] = {}

            for text, bbox, conf in filtered_results:
                # Пропускаем чисто числовые тексты — это значения, не метки
                if numeric_only.match(text.strip()):
                    continue

                # Пробуем сопоставить каждую OCR-метку с параметрами
                for param in self._param_table:
                    param_id = param.get("id")
                    if param_id is None:
                        continue

                    score = _compute_match_score(text, param)

                    # Дополнительная стратегия: word-level matching
                    # SCADA-экраны показывают сокращённые метки, а таблица содержит полные имена
                    param_name = param.get("name", "").lower()
                    param_words = set(re.findall(r'[а-яёА-ЯЁa-zA-Z]{2,}', param_name))
                    label_words = set(re.findall(r'[а-яёА-ЯЁa-zA-Z]{2,}', text.lower()))
                    common_words = param_words & label_words
                    # Если 2+ общих слова длиной >2 символов — это хороший матч
                    if len(common_words) >= 2:
                        score = min(score, 3.0)  # Word-level match
                    # Если 1 слово >4 символов — тоже считаем
                    elif any(len(w) > 4 for w in common_words):
                        score = min(score, 4.0)

                    # Дополнительно: проверяем short_name как начало или конец метки
                    short_name = param.get("short_name", "")
                    if short_name and len(short_name) >= 1:
                        text_stripped = text.strip().lower()
                        short_lower = short_name.lower()
                        if text_stripped.startswith(short_lower) or text_stripped.endswith(short_lower):
                            score = min(score, 2.0)

                    # Порог matching — чем ниже score, тем лучше (0 = идеально)
                    # Увеличили с 5.0 до 7.0 — SCADA-метки часто сокращены
                    if score > 7.0:
                        continue

                    # Если уже есть матч для этого param_id, оставляем лучший
                    if param_id in label_matches:
                        existing_score = label_matches[param_id][2]
                        if score < existing_score:
                            label_matches[param_id] = (text, bbox, score, conf)
                    else:
                        label_matches[param_id] = (text, bbox, score, conf)

            logger.debug("Fuzzy matching: %d меток сопоставлено из %d параметров",
                         len(label_matches), len(self._param_table))

            if not label_matches:
                logger.debug("Не удалось сопоставить метки с таблицей параметров")
                return {}

            # === Шаг 3: Поиск значений рядом с метками ===
            # Regex для чисел и dash-паттернов
            numeric_pattern = re.compile(r"^[+-]?\d+[.,]?\d*$")
            dash_patterns = {"---", "—", "––", "-"}
            dash_confusion = {"111", "mm", "m"}  # часто путаются с тире

            results: dict[int, str] = {}
            h, w = frame.shape[:2]

            for param_id, (label_text, label_bbox, score, label_conf) in label_matches.items():
                # Ищем значение в OCR-результатах рядом с меткой
                best_value: str | None = None
                best_value_score = float("inf")

                # Pattern A: справа от label (width * 2.5, vertical overlap)
                # Pattern B: снизу от label (height * 2.0, horizontal overlap)
                label_right = label_bbox.x + label_bbox.w
                label_bottom = label_bbox.y + label_bbox.h

                for text, bbox, conf in filtered_results:
                    # Пропускаем саму метку
                    if text == label_text:
                        continue

                    text_left = bbox.x
                    text_top = bbox.y
                    text_bottom = bbox.y + bbox.h
                    text_right = bbox.x + bbox.w

                    # Pattern A: справа от label
                    # Условия: text_left > label_right (справа)
                    #           distance < label_w * 2.5
                    #           vertical overlap
                    distance_right = text_left - label_right
                    vertical_overlap = min(label_bottom, text_bottom) - max(label_bbox.y, text_top)

                    is_pattern_a = (
                        distance_right >= 0
                        and distance_right < label_bbox.w * 2.5
                        and vertical_overlap > 0
                    )

                    # Pattern B: снизу от label
                    # Условия: text_top > label_bottom (снизу)
                    #           distance < label_h * 2.0
                    #           horizontal overlap
                    distance_below = text_top - label_bottom
                    horizontal_overlap = min(label_right, text_right) - max(label_bbox.x, text_left)

                    is_pattern_b = (
                        distance_below >= 0
                        and distance_below < label_bbox.h * 2.0
                        and horizontal_overlap > 0
                    )

                    if not (is_pattern_a or is_pattern_b):
                        continue

                    # Проверяем, является ли текст числом или dash-паттерном
                    cleaned = text.strip()
                    is_numeric = bool(numeric_pattern.match(cleaned))
                    is_dash = cleaned in dash_patterns
                    is_dash_like = cleaned in dash_confusion and conf < 0.5

                    if is_numeric:
                        # Выбираем ближайший числовой результат
                        distance = distance_right if is_pattern_a else distance_below
                        proximity_score = distance + (1.0 - conf) * 100  # штраф за низкий confidence

                        if proximity_score < best_value_score:
                            best_value = cleaned.replace(",", ".")
                            best_value_score = proximity_score

                    elif is_dash or is_dash_like:
                        # Dash-паттерн найден
                        best_value = "---"
                        best_value_score = 0  # приоритет над числами
                        break  # dash — финальный результат

                # Если значение не найдено в OCR-результатах — fallback на PaddleOCR ROI
                if best_value is None:
                    try:
                        from app.core.ocr_engine import ocr_roi_single

                        # Расширяем область справа от метки
                        expanded_x = min(label_right, w - 1)
                        expanded_w = min(int(label_bbox.w * 3), w - expanded_x)
                        expanded_y = max(0, label_bbox.y - 2)
                        expanded_h = label_bbox.h + 4

                        if expanded_w > 5 and expanded_h > 5:
                            roi_crop = frame[
                                expanded_y : expanded_y + expanded_h,
                                expanded_x : expanded_x + expanded_w,
                            ]
                            if roi_crop.size > 0:
                                roi_text, roi_conf = ocr_roi_single(roi_crop, min_confidence=0.3)
                                if roi_text and numeric_pattern.match(roi_text.strip()):
                                    best_value = roi_text.strip().replace(",", ".")
                    except ImportError:
                        logger.debug("ocr_roi_single недоступен — пропускаем fallback")
                    except Exception as e:
                        logger.debug("Ошибка OCR ROI fallback для param %d: %s", param_id, e)

                if best_value is None:
                    continue

                # === Шаг 4: Валидация и temporal smoothing ===
                # Находим параметр в таблице для получения short_name и unit
                param_info = next((p for p in self._param_table if p.get("id") == param_id), None)
                if not param_info:
                    continue

                short_name = param_info.get("short_name", "")
                unit = param_info.get("unit", "")

                # Валидируем значение
                cleaned_value, is_valid = validate_parameter_value(best_value, short_name, unit)
                if not is_valid:
                    logger.debug(
                        "Florence OCR: param %d — значение '%s' не прошло валидацию (type=%s)",
                        param_id, best_value, short_name
                    )
                    continue

                # Temporal smoothing через confidence_scorer
                if self._confidence_scorer:
                    try:
                        smoothed_value, smoothed_conf = self._confidence_scorer.smooth_value(
                            param_id=param_id,
                            current_value=cleaned_value,
                            current_confidence=label_conf,
                        )
                        if smoothed_value is not None:
                            cleaned_value = smoothed_value
                    except Exception as e:
                        logger.debug("Ошибка smoothing для param %d: %s", param_id, e)

                results[param_id] = cleaned_value
                logger.debug(
                    "Florence OCR: param %d = '%s' (label='%s', score=%.1f)",
                    param_id, cleaned_value, label_text, score
                )

            return results

        except Exception as e:
            logger.warning("Ошибка Florence-2 распознавания: %s", e, exc_info=True)
            return {}

    def _recognize_frame_with_logging(
        self,
        frame: np.ndarray,
        zones: list,
        ocr_log: OcrLogger,
        crop_offset: tuple[int, int] = (0, 0),
        full_frame_shape: tuple[int, int] | None = None,
        full_frame: np.ndarray | None = None,
    ) -> dict[int, str]:
        """Распознаёт значения параметров на кадре с логированием.

        Использует Enhanced pipeline (Paddle + Florence + Layout Analysis)
        для получения пар label:value и отдельных OCR-результатов.

        Стратегия (по приоритету):
        1. Enhanced pair matching — прямое сопоставление пар из Layout Analysis
           с метками параметров из калибровки (short_name)
        2. Proximity search — поиск значения рядом с label_bbox
        3. ROI filtering — фильтрация OCR результатов по roi_bbox/value_bbox

        Args:
            frame: Предобработанный кадр (может быть обрезан).
            zones: Зоны мнемосхемы.
            ocr_log: OCR логгер для записи результатов.
            crop_offset: Смещение (x, y) если кадр был обрезан до центральной зоны.
            full_frame_shape: Размеры полного кадра (h, w) для нормализации координат.
            full_frame: Полный кадр (не обрезанный) для цветовой классификации.

        Returns:
            Словарь param_id -> отформатированное значение.
        """
        from app.core.ocr_engine import ocr_full_frame_enhanced
        from app.core.ocr_models import TextBox, TextPair

        # Если нет калибровки или привязок — используем fullframe fallback
        if not self._calibration or not self._calibration.mappings:
            return self._fullframe_fallback(frame)

        self._frame_counter += 1
        frame_start = time.perf_counter()

        params: dict[int, str] = {}
        value_processing_ms = 0.0
        # Отслеживаем параметры с реальным OCR-распознаванием на этом кадре
        detected_param_ids: set[int] = set()

        h, w = frame.shape[:2]
        # Используем размеры полного кадра для нормализации если предоставлены
        frame_h, frame_w = full_frame_shape if full_frame_shape is not None else (h, w)
        x_offset, y_offset = crop_offset

        # === Enhanced OCR: Paddle + Florence + Layout Analysis ===
        enhanced_t0 = time.perf_counter()
        enhanced_result = ocr_full_frame_enhanced(frame)
        enhanced_ms = (time.perf_counter() - enhanced_t0) * 1000

        # Извлекаем пары label:value из Enhanced pipeline
        enhanced_pairs: list[TextPair] = []
        if hasattr(enhanced_result, 'pairs') and enhanced_result.pairs:
            # Фильтруем пары с некорректными label/value типами
            from app.core.ocr_models import TextBox as _TB
            for p in enhanced_result.pairs:
                if isinstance(getattr(p, 'label', None), _TB) and isinstance(getattr(p, 'value', None), _TB):
                    enhanced_pairs.append(p)
                else:
                    logger.debug(
                        "Пропуск пары: label=%s, value=%s",
                        type(getattr(p, 'label', None)).__name__,
                        type(getattr(p, 'value', None)).__name__,
                    )

        # === d) Корректировка координат после обрезки ===
        if crop_offset != (0, 0) and enhanced_pairs:
            for pair in enhanced_pairs:
                if pair.label:
                    pair.label.bbox.x += x_offset
                    pair.label.bbox.y += y_offset
                if pair.value:
                    pair.value.bbox.x += x_offset
                    pair.value.bbox.y += y_offset

        # === e) Цветовая классификация текстовых блоков (AFTER OCR) ===
        try:
            color_t0 = time.perf_counter()
            # Собираем все TextBox из пар для классификации
            all_text_boxes: list[TextBox] = []
            for pair in enhanced_pairs:
                if pair.label:
                    all_text_boxes.append(pair.label)
                if pair.value:
                    all_text_boxes.append(pair.value)

            # Классифицируем по цвету на полном кадре (не обрезанном)
            if all_text_boxes:
                color_frame = full_frame if full_frame is not None else frame
                self._color_filter.classify_textboxes(color_frame, all_text_boxes)
            color_ms = (time.perf_counter() - color_t0) * 1000
            ocr_log.log_stage("color_classification", color_ms)
        except Exception as e:
            logger.warning("Ошибка цветовой классификации: %s", e)

        # === g) Confidence scoring для пар ===
        try:
            score_t0 = time.perf_counter()
            scored_pairs: list[tuple[TextPair, float]] = []
            for pair in enhanced_pairs:
                # Скорим value бокс с контекстом пары
                score = self._confidence_scorer.score(
                    text_box=pair.value,
                    pair=pair,
                    param_range=None,
                    recent_values=None,
                )
                scored_pairs.append((pair, score))

            # Фильтруем пары с confidence < threshold
            filtered_pairs = [
                pair for pair, score in scored_pairs
                if score >= self._confidence_threshold
            ]
            if filtered_pairs:
                enhanced_pairs = filtered_pairs

            score_ms = (time.perf_counter() - score_t0) * 1000
            ocr_log.log_stage("confidence_scoring", score_ms)
            logger.debug(
                "Confidence scoring: %d пар отфильтровано до %d (порог=%.2f)",
                len(scored_pairs), len(enhanced_pairs), self._confidence_threshold
            )
        except Exception as e:
            logger.warning("Ошибка confidence scoring: %s", e)

        # Конвертируем TextBox из Enhanced pipeline в OCRTextResult
        # для proximity search и ROI filtering
        all_ocr_results = self._enhanced_to_ocr_results(enhanced_result, frame_h, frame_w)

        logger.debug(
            "Enhanced OCR: %d текстов, %d пар (%.1f мс)",
            len(all_ocr_results), len(enhanced_pairs), enhanced_ms
        )

        # Строим индекс: short_name → (value_text, confidence) из Enhanced pairs
        enhanced_label_map: dict[str, tuple[str, float]] = {}
        for pair in enhanced_pairs:
            label_text = pair.label.text.strip() if pair.label and pair.label.text else ""
            value_text = pair.value.text.strip() if pair.value and pair.value.text else ""
            if label_text and value_text:
                enhanced_label_map[label_text] = (value_text, pair.pair_confidence)

        for mapping in self._calibration.mappings:
            value_text: str | None = None
            confidence: float = 0.0
            detection_method: str = "none"

            # === Приоритет 0: Enhanced pair matching ===
            # Прямое сопоставление пар из Layout Analysis с short_name параметра
            if mapping.short_name and mapping.short_name in enhanced_label_map:
                val, conf = enhanced_label_map[mapping.short_name]
                value_text = val
                confidence = conf
                detection_method = "enhanced_pair"
                logger.debug(
                    "Param %d (%s): found '%s' (conf %.2f) via Enhanced pair",
                    mapping.param_id, mapping.short_name, value_text[:20], confidence
                )

            # === Приоритет 1: Proximity search (если есть label_bbox) ===
            if value_text is None and mapping.label_bbox is not None:
                proximity_t0 = time.perf_counter()
                number, conf, position = self._find_value_near_label(
                    mapping.label_bbox,
                    all_ocr_results,
                    frame.shape[:2],
                )
                proximity_ms = (time.perf_counter() - proximity_t0) * 1000

                if number is not None:
                    # Значение найдено через proximity search
                    value_text = str(number)
                    confidence = conf
                    detection_method = f"proximity_{position}"
                    logger.debug(
                        "Param %d: found %.1f (conf %.2f) %s of label (%.1f ms)",
                        mapping.param_id, number, confidence, position, proximity_ms
                    )

            # === Приоритет 2: ROI filtering (если proximity не сработал и есть roi_bbox/value_bbox) ===
            if value_text is None and (mapping.value_bbox or mapping.roi_bbox):
                roi_to_use = mapping.value_bbox if mapping.value_bbox else mapping.roi_bbox
                # Фильтруем OCR результаты по ROI — БЕЗ дополнительного OCR вызова
                roi_text, roi_conf = self._filter_ocr_by_roi(all_ocr_results, roi_to_use, (h, w))
                if roi_text is not None:
                    value_text = roi_text
                    confidence = roi_conf
                    detection_method = "roi_filter"
                    logger.debug(
                        "Param %d: found '%s' (conf %.2f) via ROI filter",
                        mapping.param_id, value_text[:20], confidence
                    )

            # === Обработка результата ===
            if value_text is not None:
                detected_param_ids.add(mapping.param_id)

                # Цветовое состояние (по value_bbox если есть, иначе по label_bbox)
                color_state_bbox = mapping.value_bbox if mapping.value_bbox else mapping.label_bbox
                color_state = detect_color_state(frame, color_state_bbox) if color_state_bbox else "normal"

                # Обработка значения
                vp_t0 = time.perf_counter()
                processed_value = self._value_processor.process_value(
                    mapping.param_id,
                    value_text,
                    confidence,
                    mapping.short_name,
                    color_state,
                    detection_method=detection_method,
                )
                value_processing_ms += (time.perf_counter() - vp_t0) * 1000
                params[mapping.param_id] = processed_value
            else:
                # Параметр не найден — проверяем temporal smoothing
                if mapping.param_id in self._value_processor._last_valid:
                    # Используем последнее валидное значение
                    params[mapping.param_id] = self._value_processor.process_value(
                        mapping.param_id, "", 0.0, mapping.short_name
                    )
                    logger.debug("Param %d: using cached value (temporal smoothing)", mapping.param_id)

        # Логируем стадии OCR и обработки значений
        total_ms = (time.perf_counter() - frame_start) * 1000
        ocr_log.log_stage("ocr_enhanced", enhanced_ms)
        ocr_log.log_stage("value_processing", value_processing_ms)

        logger.debug(
            "Кадр: %d параметров (%d детектировано, %d из _last_valid) за %.1f мс",
            len(params), len(detected_param_ids), len(params) - len(detected_param_ids), total_ms
        )

        return params

    def _enhanced_to_ocr_results(
        self,
        enhanced_result,
        frame_h: int,
        frame_w: int,
    ) -> list[OCRTextResult]:
        """Конвертирует результат Enhanced pipeline в список OCRTextResult.

        Извлекает TextBox из пар (label + value) и преобразует
        BBox (пиксели) в BoundingBox (нормализованные координаты),
        чтобы результаты можно было использовать в proximity search
        и ROI filtering.

        Args:
            enhanced_result: RecognitionResult от OcrPipeline.
            frame_h: Высота кадра.
            frame_w: Ширина кадра.

        Returns:
            Список OCRTextResult с нормализованными координатами.
        """
        from app.core.ocr_models import TextBox

        ocr_results: list[OCRTextResult] = []
        seen_texts: set[str] = set()  # Для дедупликации

        # Извлекаем тексты из пар (label + value)
        if hasattr(enhanced_result, 'pairs') and enhanced_result.pairs:
            for pair in enhanced_result.pairs:
                for text_box in (pair.label, pair.value):
                    if not isinstance(text_box, TextBox):
                        continue
                    if text_box and text_box.text and text_box.text.strip():
                        # Дедупликация по тексту + примерным координатам
                        dedup_key = f"{text_box.text.strip()}@{text_box.bbox.x // 10},{text_box.bbox.y // 10}"
                        if dedup_key in seen_texts:
                            continue
                        seen_texts.add(dedup_key)

                        bbox = text_box.bbox
                        # BBox (pixel x,y,w,h) → BoundingBox (normalized x1,y1,x2,y2)
                        norm_bbox = BoundingBox(
                            x1=bbox.x / max(frame_w, 1),
                            y1=bbox.y / max(frame_h, 1),
                            x2=(bbox.x + bbox.w) / max(frame_w, 1),
                            y2=(bbox.y + bbox.h) / max(frame_h, 1),
                        )
                        ocr_results.append(OCRTextResult(
                            text=text_box.text.strip(),
                            confidence=text_box.confidence,
                            bbox=norm_bbox,
                        ))

        return ocr_results

    def _fullframe_fallback(self, frame: np.ndarray) -> dict[int, str]:
        """Полный OCR без калибровки — возвращает {idx: value} из пар label-value."""
        from app.core.ocr_engine import ocr_full_frame, ocr_full_frame_enhanced
        from app.core.spatial_clusterer import cluster_label_value_pairs
        from app.core.ocr_models import RecognitionResult

        # Пробуем enhanced pipeline для пар
        enhanced_result = ocr_full_frame_enhanced(frame)
        if isinstance(enhanced_result, RecognitionResult) and enhanced_result.pairs:
            # Используем пары из enhanced pipeline
            pairs = cluster_label_value_pairs(
                self._enhanced_to_ocr_results(enhanced_result, frame.shape[0], frame.shape[1]),
                frame.shape[:2],
            )
            if pairs:
                return {i + 1: p.value for i, p in enumerate(pairs) if p.value}

        # Fallback: прямой PaddleOCR
        results = ocr_full_frame(frame)
        if not results:
            return {}

        pairs = cluster_label_value_pairs(results, frame.shape[:2])
        return {i + 1: p.value for i, p in enumerate(pairs) if p.value}

    def _recognize_frame_ocr_only(
        self,
        frame: np.ndarray,
        ocr_log: OcrLogger,
    ) -> dict[int, str]:
        """Распознаёт параметры в OCR-only режиме (без калибровки).

        Использует двухпутевой OCR пайплайн (Paddle + Florence) для
        извлечения всех текстовых пар label:value на кадре.
        Fallback к прямому PaddleOCR если пар не найдено.

        Args:
            frame: Предобработанный кадр.
            ocr_log: OCR логгер для записи результатов.

        Returns:
            Словарь param_id -> отформатированное значение.
        """
        from app.core.ocr_engine import ocr_full_frame_enhanced, ocr_full_frame
        from app.core.ocr_models import RecognitionResult

        params: dict[int, str] = {}

        # Стратегия 1: Enhanced pipeline (Paddle + Layout + Florence)
        ocr_t0 = time.perf_counter()
        result = ocr_full_frame_enhanced(frame)
        ocr_ms = (time.perf_counter() - ocr_t0) * 1000

        raw_fields = result.raw_fields if hasattr(result, 'raw_fields') else {}
        confidence = result.confidence if hasattr(result, 'confidence') else 0.0
        source = result.source if hasattr(result, 'source') else "paddle"

        # Логируем пары из enhanced pipeline
        if hasattr(result, 'pairs') and result.pairs:
            ocr_log.log_pairs(result.pairs)

        # Стратегия 2: Fallback — прямой PaddleOCR, если пар не найдено
        fallback_reason = None
        if not raw_fields:
            fallback_reason = "enhanced_pipeline_no_pairs"
            logger.info("OCR-only: 0 пар от enhanced, fallback -> прямой PaddleOCR")
            ocr_log.log_fallback(reason=fallback_reason)

            fallback_t0 = time.perf_counter()
            ocr_results = ocr_full_frame(frame)
            fallback_ms = (time.perf_counter() - fallback_t0) * 1000
            ocr_log.log_stage("fallback_paddle", fallback_ms, None)

            if ocr_results:
                ocr_log.log_fallback(ocr_results, reason=fallback_reason)
                for r in ocr_results:
                    if r.text.strip():
                        raw_fields[f"text_{len(raw_fields)}"] = r.text.strip()
                source = "paddle_fallback"
                logger.info("Fallback: распознано %d текстов", len(ocr_results))
            else:
                logger.warning("PaddleOCR fallback: 0 результатов")

        # Формируем параметры из raw_fields
        for p_idx, (label, value) in enumerate(raw_fields.items(), 1):
            params[p_idx] = value

        # Логируем результат как RecognitionResult
        rec_result = RecognitionResult(
            raw_fields={pid: str(val) for pid, val in params.items()},
            confidence=confidence,
            source=source,
            pairs=result.pairs if hasattr(result, 'pairs') else [],
            frame_idx=0,
            processing_ms=ocr_ms,
        )
        ocr_log.log_result(rec_result, is_duplicate=False, fallback=fallback_reason)
        ocr_log.log_stage("ocr_full_frame", ocr_ms)

        logger.info("OCR-only: извлечено %d параметров (source=%s)", len(params), source)
        return params
