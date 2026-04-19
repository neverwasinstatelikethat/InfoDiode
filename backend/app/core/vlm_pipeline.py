"""Основной пайплайн обработки видео через VLM Qwen3.5-4B.

Модуль реализует полный цикл обработки видео:
- Извлечение кадров с интервалом 500мс
- Анализ через VLM (Vision-Language Model)
- Нормализация и валидация параметров
- Генерация XML в формате <sheme>
- Шифрование GPG и отправка email
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from pydantic import ValidationError

from app.config import Settings, settings
from app.core.crypto_service import encrypt_xml
from app.core.email_service import send_xml_email
from app.core.prompts import (
    ParameterType,
    build_messages,
    build_residual_messages,
    build_zone_messages,
    validate_value_in_range,
)
from app.core.vlm_client import VLMClient, get_vlm_client
from app.core.xml_generator import create_snapshot, generate_xml
from app.core.zone_segmentor import SCADA_ZONES, ZoneCrop, segment_frame
from app.models.schemas import VLMParameter, VLMResponse

logger = logging.getLogger(__name__)

# Тип callback-функции для отчётов о прогрессе
ProgressCallback = Callable[[str, float, str, int, int], None]


@dataclass
class PipelineResult:
    """Результат обработки видео через VLM пайплайн.

    Attributes:
        video_id: Идентификатор видео.
        xml_path: Путь к сгенерированному XML файлу.
        encrypted_path: Путь к зашифрованному файлу.
        total_frames: Общее количество кадров в видео.
        processed_frames: Количество успешно обработанных кадров.
        failed_frames: Количество кадров с ошибками.
        total_parameters: Общее количество извлечённых параметров.
        processing_time_seconds: Время обработки в секундах.
        status: Статус выполнения ("completed" | "partial" | "failed").
        errors: Список сообщений об ошибках.
    """

    video_id: str
    xml_path: Path | None = None
    encrypted_path: Path | None = None
    total_frames: int = 0
    processed_frames: int = 0
    failed_frames: int = 0
    total_parameters: int = 0
    processing_time_seconds: float = 0.0
    status: str = "pending"
    errors: list[str] = field(default_factory=list)


class VLMPipeline:
    """Основной пайплайн обработки видео через VLM.

    Реализует параллельную обработку кадров видео с использованием
    Vision-Language Model для извлечения параметров SCADA мнемосхем.

    Attributes:
        settings: Настройки приложения.
        vlm_client: Клиент для взаимодействия с VLM.
        frame_interval_ms: Интервал извлечения кадров в миллисекундах.
        use_clahe: Использовать ли CLAHE для предобработки.
        skip_similar_frames: Пропускать ли похожие кадры.
        similarity_threshold: Порог схожести для пропуска.
        concurrency: Количество параллельных VLM запросов.
    """

    def __init__(
        self,
        settings_obj: Settings | None = None,
        vlm_client: VLMClient | None = None,
    ) -> None:
        """Инициализирует пайплайн с заданными параметрами.

        Args:
            settings_obj: Настройки приложения (если None, используются глобальные).
            vlm_client: Клиент VLM (если None, создаётся из настроек).
        """
        self.settings = settings_obj or settings
        self.vlm_client = vlm_client or get_vlm_client()
        self.frame_interval_ms = self.settings.vlm_frame_interval_ms
        self.use_clahe = True  # CLAHE включён по умолчанию для улучшения контраста
        # Оптимизации производительности
        self.skip_similar_frames = getattr(self.settings, "vlm_skip_similar_frames", True)
        self.similarity_threshold = getattr(self.settings, "vlm_similarity_threshold", 0.95)
        self.concurrency = getattr(self.settings, "vlm_concurrency", 2)
        # Zone-based analysis settings
        self.zone_enabled = getattr(self.settings, "vlm_zone_enabled", True)
        self.zone_crop_padding = getattr(self.settings, "vlm_zone_crop_padding_px", 15)
        self.zone_min_crop_size = getattr(self.settings, "vlm_zone_min_crop_size", 512)
        # Кэш предыдущего кадра для сравнения
        self._prev_frame_gray: np.ndarray | None = None
        self._prev_result: dict[str, Any] | None = None

        logger.info(
            "VLMPipeline инициализирован: interval=%dms, clahe=%s, skip_similar=%s, "
            "similarity_thresh=%.2f, concurrency=%d, zone_enabled=%s",
            self.frame_interval_ms,
            self.use_clahe,
            self.skip_similar_frames,
            self.similarity_threshold,
            self.concurrency,
            self.zone_enabled,
        )

    async def process_video(
        self,
        video_path: Path,
        video_id: str,
        send_email: bool = True,
        progress_callback: ProgressCallback | None = None,
        parameter_table: list[dict] | None = None,
    ) -> PipelineResult:
        """Выполняет полный цикл обработки видео.

        Args:
            video_path: Путь к видеофайлу.
            video_id: Идентификатор видео.
            send_email: Отправить зашифрованный XML по email.
            progress_callback: Функция для отчётов о прогрессе.
            parameter_table: Пользовательская таблица параметров из xlsx/csv.

        Returns:
            PipelineResult с результатами обработки.
        """
        start_time = time.monotonic()
        result = PipelineResult(video_id=video_id)

        # INFO: Pipeline start with full settings summary
        logger.info(
            "Pipeline START: video_id=%s, video_path=%s, model=%s, temperature=%.2f, "
            "max_tokens=%d, frame_interval=%dms, clahe=%s",
            video_id,
            video_path,
            self.vlm_client.model_name,
            self.vlm_client.temperature,
            self.vlm_client.max_tokens,
            self.frame_interval_ms,
            self.use_clahe,
        )
        logger.info("Начало обработки видео: %s (ID: %s)", video_path, video_id)

        try:
            # 1. Проверка существования видео
            if not video_path.exists():
                raise FileNotFoundError(f"Видеофайл не найден: {video_path}")

            # 2. Извлечение кадров
            if progress_callback:
                progress_callback(video_id, 0.0, "frame_extraction", 0, 0)

            frames = self.extract_frames(video_path)
            result.total_frames = len(frames)

            if not frames:
                raise RuntimeError("Не удалось извлечь кадры из видео")

            # 3. Анализ кадров через VLM
            if progress_callback:
                progress_callback(video_id, 5.0, "vlm_analysis", 0, len(frames))

            raw_results = await self.analyze_frames(
                frames,
                video_id=video_id,
                progress_callback=progress_callback,
                parameter_table=parameter_table,
            )

            # Подсчёт успешных/неуспешных кадров
            for r in raw_results:
                if "error" in r:
                    result.failed_frames += 1
                else:
                    result.processed_frames += 1
                    # Подсчёт параметров
                    params = r.get("parameters", [])
                    result.total_parameters += len(params)

            # 4. Нормализация параметров
            if progress_callback:
                progress_callback(video_id, 85.0, "normalization", len(frames), len(frames))

            normalized_results = self.normalize_parameters(raw_results)

            # 5. Временное сглаживание
            smoothed_results = self.temporal_smoothing(normalized_results)

            # 5.5. Нормализация лейблов через fuzzy matching с таблицей
            if progress_callback:
                progress_callback(video_id, 87.5, "label_normalization", len(frames), len(frames))
            
            smoothed_results = self._normalize_parameter_labels(
                smoothed_results,
                parameter_table=parameter_table,
                threshold=0.80,
            )

            # 6. Генерация XML
            if progress_callback:
                progress_callback(video_id, 90.0, "xml_generation", len(frames), len(frames))

            xml_content, xml_path = self._generate_xml_output(
                smoothed_results,
                video_id,
                parameter_table=parameter_table,
            )
            result.xml_path = xml_path

            # 7. Шифрование GPG
            if progress_callback:
                progress_callback(video_id, 95.0, "encryption", len(frames), len(frames))

            # INFO: GPG encryption before
            logger.debug(
                "GPG encryption: input_path=%s, input_size=%.1fKB",
                xml_path,
                len(xml_content.encode("utf-8")) / 1024,
            )

            encrypted_bytes = encrypt_xml(xml_content)
            encrypted_path = xml_path.with_suffix(".xml.gpg")
            encrypted_path.write_bytes(encrypted_bytes)
            result.encrypted_path = encrypted_path

            # INFO: GPG encryption success
            logger.info(
                "GPG encryption: input=%s, output=%s, encrypted_size=%.1fKB, success=True",
                xml_path,
                encrypted_path,
                len(encrypted_bytes) / 1024,
            )
            logger.info("XML зашифрован: %s", encrypted_path)

            # 8. Отправка email (опционально)
            if send_email:
                if progress_callback:
                    progress_callback(video_id, 98.0, "email_sending", len(frames), len(frames))

                # INFO: Email dispatch
                logger.info(
                    "Email dispatch: recipient=%s, attachment=%s, attachment_size=%.1fKB",
                    self.settings.smtp_recipient,
                    f"{video_id}_output.xml.gpg",
                    len(encrypted_bytes) / 1024,
                )

                email_success = send_xml_email(
                    encrypted_data=encrypted_bytes,
                    filename=f"{video_id}_output.xml.gpg",
                    subject=f"InfoDiode: SCADA Data from {video_id}",
                )
                if email_success:
                    logger.info(
                        "Email dispatch: success=True, recipient=%s, attachment_size=%.1fKB",
                        self.settings.smtp_recipient,
                        len(encrypted_bytes) / 1024,
                    )
                else:
                    logger.error(
                        "Email dispatch: success=False, recipient=%s, attachment_size=%.1fKB",
                        self.settings.smtp_recipient,
                        len(encrypted_bytes) / 1024,
                    )
                    result.errors.append("Не удалось отправить email")

            # Определение статуса
            if result.failed_frames == 0:
                result.status = "completed"
            elif result.failed_frames / result.total_frames > 0.5:
                result.status = "failed"
            else:
                result.status = "partial"

        except FileNotFoundError as e:
            result.status = "failed"
            result.errors.append(str(e))
            logger.error("Файл не найден: %s", e)

        except Exception as e:
            result.status = "failed"
            result.errors.append(f"Ошибка обработки: {str(e)}")
            logger.exception("Ошибка обработки видео: %s", e)

        finally:
            result.processing_time_seconds = time.monotonic() - start_time

            if progress_callback:
                progress_callback(
                    video_id,
                    100.0,
                    f"completed_{result.status}",
                    result.processed_frames,
                    result.total_frames,
                )

        # INFO: Pipeline completion summary
        logger.info(
            "Pipeline COMPLETE: video_id=%s, status=%s, elapsed_time=%.2fs, "
            "frames_processed=%d, frames_failed=%d, total_frames=%d, total_params=%d",
            video_id,
            result.status,
            result.processing_time_seconds,
            result.processed_frames,
            result.failed_frames,
            result.total_frames,
            result.total_parameters,
        )
        logger.info(
            "Обработка завершена: status=%s, frames=%d/%d, params=%d, time=%.2fs",
            result.status,
            result.processed_frames,
            result.total_frames,
            result.total_parameters,
            result.processing_time_seconds,
        )

        return result

    def extract_frames(self, video_path: Path) -> list[tuple[np.ndarray, str]]:
        """Извлекает кадры из видео с интервалом 500мс.

        Кадры извлекаются в моменты времени .001 и .501 секунды
        для соответствия спецификации проекта.

        Args:
            video_path: Путь к видеофайлу.

        Returns:
            Список кортежей (frame, timestamp), где:
            - frame: numpy array (BGR формат)
            - timestamp: строка в формате HH:MM:SS.mmm

        Raises:
            RuntimeError: Если не удалось открыть видеофайл.
        """
        frames: list[tuple[np.ndarray, str]] = []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = int(total_frames / fps * 1000) if fps > 0 else 0

            # INFO: Frame extraction details
            frames_to_extract = duration_ms // self.frame_interval_ms if self.frame_interval_ms > 0 else 0
            logger.info(
                "Frame extraction: video_fps=%.2f, duration=%.2fs (%dms), total_video_frames=%d, "
                "frames_to_extract=%d, interval=%dms",
                fps,
                duration_ms / 1000.0,
                duration_ms,
                total_frames,
                frames_to_extract,
                self.frame_interval_ms,
            )

            logger.debug(
                "Видео: fps=%.2f, total_frames=%d, duration=%dms",
                fps,
                total_frames,
                duration_ms,
            )

            # Извлекаем кадры каждые 500мс
            # Timestamps: .001, .501, 1.001, 1.501, ... (в секундах)
            interval_ms = self.frame_interval_ms  # 500мс по умолчанию

            current_ms = 1  # Начинаем с 1мс (получим .001)
            frame_count = 0

            while current_ms < duration_ms:
                # Позиционируемся на нужный кадр
                cap.set(cv2.CAP_PROP_POS_MSEC, current_ms)

                ret, frame = cap.read()
                if not ret:
                    logger.warning("Не удалось прочитать кадр на %dms", current_ms)
                    current_ms += interval_ms
                    continue

                # Формируем timestamp в формате HH:MM:SS.mmm
                timestamp = self._format_timestamp_ms(current_ms)
                frames.append((frame.copy(), timestamp))

                # Опционально применяем CLAHE для улучшения контраста
                if self.use_clahe:
                    frames[-1] = (self._apply_clahe(frames[-1][0]), frames[-1][1])

                frame_count += 1
                current_ms += interval_ms

            logger.info(
                "Извлечено %d кадров из %s (интервал %dms)",
                frame_count,
                video_path.name,
                interval_ms,
            )
            logger.info("Frame extraction complete: %d frames extracted", frame_count)

        finally:
            cap.release()

        return frames

    def _compute_frame_similarity(self, frame: np.ndarray) -> float:
        """Вычисляет схожесть текущего кадра с предыдущим.
    
        Использует нормализованную корреляцию (NCC) для быстрого сравнения.
        Значение 1.0 = идентичные кадры, 0.0 = полностью разные.
    
        Args:
            frame: Текущий кадр в формате BGR.
    
        Returns:
            Коэффициент схожести от 0.0 до 1.0.
        """
        # Конвертируем в grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            return 0.0  # Первый кадр — не похож ни на что
    
        # Вычисляем нормализованную корреляцию
        # Результат от -1 до 1, нормализуем к 0-1
        result = cv2.matchTemplate(
            gray, self._prev_frame_gray, cv2.TM_CCOEFF_NORMED
        )
        similarity = float(result[0, 0])
        # Нормализуем от [-1, 1] к [0, 1]
        similarity = (similarity + 1.0) / 2.0
    
        return similarity
    
    async def _analyze_single_frame(
        self,
        frame: np.ndarray,
        timestamp: str,
        frame_idx: int,
        total_frames: int,
        parameter_table: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Анализирует один кадр: зонный или полный анализ.

        Если zone_enabled=True, разбивает кадр на зоны и анализирует
        каждую зону отдельным VLM-запросом с zone-specific промптом.
        Иначе — fallback на полный кадр (старое поведение).

        Args:
            frame: Изображение в формате BGR.
            timestamp: Временная метка кадра.
            frame_idx: Индекс кадра (для логирования).
            total_frames: Общее количество кадров.
            parameter_table: Пользовательская таблица параметров.

        Returns:
            Результат анализа с timestamp.
        """
        if self.zone_enabled:
            return await self._analyze_frame_by_zones(
                frame, timestamp, frame_idx, total_frames,
                parameter_table=parameter_table,
            )
        # Fallback: полный анализ кадра (старое поведение)
        return await self._analyze_full_frame(
            frame, timestamp, frame_idx, total_frames,
            parameter_table=parameter_table,
        )

    async def _analyze_frame_by_zones(
        self,
        frame: np.ndarray,
        timestamp: str,
        frame_idx: int,
        total_frames: int,
        parameter_table: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Зонный анализ кадра: параллельные запросы по зонам + residual.

        1. Сегментирует кадр на зоны через zone_segmentor
        2. Для каждой зоны формирует zone-specific промпт
        3. Параллельно отправляет запросы (semaphore-controlled)
        4. Анализирует residual области (multi-image)
        5. Мержит результаты с дедупликацией по overlap

        Args:
            frame: Изображение в формате BGR.
            timestamp: Временная метка кадра.
            frame_idx: Индекс кадра.
            total_frames: Общее количество кадров.
            parameter_table: Пользовательская таблица параметров.

        Returns:
            Мерженный результат анализа со всеми параметрами.
        """
        frame_start = time.perf_counter()

        try:
            # 1. Сегментация кадра
            zone_crops, residual_crops = segment_frame(
                frame,
                zones=SCADA_ZONES,
                padding=self.zone_crop_padding,
                min_crop_size=self.zone_min_crop_size,
            )

            logger.info(
                "Frame %d/%d (%s): zone_segmentation=%d zones + %d residual",
                frame_idx + 1, total_frames, timestamp,
                len(zone_crops), len(residual_crops),
            )

            # 2. Параллельный анализ зон
            semaphore = asyncio.Semaphore(self.concurrency)
            zone_results: list[dict[str, Any]] = []

            async def _analyze_zone(zc: ZoneCrop) -> dict[str, Any]:
                """Анализирует одну зону с семафором."""
                async with semaphore:
                    system_prompt, user_prompt = build_zone_messages(
                        zone_id=zc.zone_id,
                        zone_name=zc.zone_name,
                        frame_timestamp=timestamp,
                    )
                    logger.info(
                        "Frame %d/%d (%s): Zone %d (%s) — calling VLM...",
                        frame_idx + 1, total_frames, timestamp,
                        zc.zone_id, zc.zone_name,
                    )
                    result = await self.vlm_client.analyze_frame(
                        frame=zc.crop,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                    # Добавляем зону в результат
                    if isinstance(result, dict):
                        result["zone_id"] = zc.zone_id
                    return result

            # Запускаем все зоны параллельно
            zone_tasks = [_analyze_zone(zc) for zc in zone_crops]
            zone_task_results = await asyncio.gather(*zone_tasks, return_exceptions=True)

            for r in zone_task_results:
                if isinstance(r, Exception):
                    logger.error(
                        "Frame %d/%d (%s): Zone analysis failed — %s",
                        frame_idx + 1, total_frames, timestamp, str(r)[:300],
                    )
                    zone_results.append({"error": str(r), "parameters": [], "zone_id": -1})
                else:
                    zone_results.append(r)

            # 3. Анализ residual областей (multi-image)
            residual_params: list[dict] = []
            if residual_crops:
                try:
                    res_system, res_user = build_residual_messages(timestamp)
                    logger.info(
                        "Frame %d/%d (%s): Residual analysis (%d crops)...",
                        frame_idx + 1, total_frames, timestamp, len(residual_crops),
                    )
                    residual_result = await self.vlm_client.analyze_multi_image(
                        frames=residual_crops,
                        system_prompt=res_system,
                        user_prompt=res_user,
                    )
                    residual_params = residual_result.get("parameters", [])
                    logger.info(
                        "Frame %d/%d (%s): Residual — %d params found",
                        frame_idx + 1, total_frames, timestamp, len(residual_params),
                    )
                except Exception as e:
                    logger.warning(
                        "Frame %d/%d (%s): Residual analysis failed — %s",
                        frame_idx + 1, total_frames, timestamp, str(e)[:200],
                    )

            # 4. Мержим результаты
            merged = self._merge_zone_results(
                zone_results=zone_results,
                residual_params=residual_params,
                timestamp=timestamp,
                frame_idx=frame_idx,
                total_frames=total_frames,
            )

            elapsed_ms = (time.perf_counter() - frame_start) * 1000
            logger.info(
                "Frame %d/%d (%s): ZONE analysis COMPLETE %.1fms — total_params=%d, zones=%d",
                frame_idx + 1, total_frames, timestamp,
                elapsed_ms, len(merged.get("parameters", [])), len(zone_crops),
            )

            return merged

        except Exception as e:
            elapsed_ms = (time.perf_counter() - frame_start) * 1000
            logger.error(
                "Frame %d/%d (%s): ZONE analysis FAILED after %.1fms — %s: %s",
                frame_idx + 1, total_frames, timestamp,
                elapsed_ms, type(e).__name__, str(e)[:500],
            )
            return {
                "timestamp": timestamp,
                "error": str(e),
                "parameters": [],
                "frame_quality": "error",
            }

    def _merge_zone_results(
        self,
        zone_results: list[dict[str, Any]],
        residual_params: list[dict],
        timestamp: str,
        frame_idx: int,
        total_frames: int,
    ) -> dict[str, Any]:
        """Мержит результаты анализа зон + residual в один результат.

        Дедупликация: если параметр найден в нескольких зонах (overlap),
        оставляем с более высоким confidence.

        Args:
            zone_results: Результаты от каждой зоны.
            residual_params: Параметры из residual областей.
            timestamp: Временная метка кадра.
            frame_idx: Индекс кадра.
            total_frames: Общее количество кадров.

        Returns:
            Мерженный результат с полным списком параметров.
        """
        all_params: list[dict] = []

        # Собираем параметры из зон
        for zr in zone_results:
            if "error" in zr and "parameters" not in zr:
                continue
            params = zr.get("parameters", [])
            zone_id = zr.get("zone_id", -1)
            for p in params:
                p["zone_id"] = zone_id
                all_params.append(p)

        # Добавляем параметры из residual
        for p in residual_params:
            p["zone_id"] = 0
            all_params.append(p)

        # Дедупликация по label (overlap зон)
        seen: dict[str, dict] = {}  # label -> best param
        for p in all_params:
            label = p.get("label", "")
            if not label:
                continue
            # Нормализуем для сравнения
            key = label.strip().lower()
            confidence = float(p.get("confidence", 0.5))
            if key not in seen or confidence > float(seen[key].get("confidence", 0.5)):
                seen[key] = p

        deduped_params = list(seen.values())

        # Валидация через Pydantic
        try:
            vlm_response = VLMResponse.model_validate({"parameters": deduped_params})
            deduped_params = [p.model_dump() for p in vlm_response.parameters]
        except ValidationError as e:
            logger.warning(
                "Frame %d/%d: Pydantic validation on merged results failed — %s",
                frame_idx + 1, total_frames, str(e)[:300],
            )

        return {
            "timestamp": timestamp,
            "parameters": deduped_params,
            "mnemonic_id": "",  # Определяется позже при label normalization
            "frame_quality": "good",
            "total_params_found": len(deduped_params),
            "zones_processed": len(zone_results),
        }

    async def _analyze_full_frame(
        self,
        frame: np.ndarray,
        timestamp: str,
        frame_idx: int,
        total_frames: int,
        parameter_table: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Анализирует полный кадр через VLM (fallback, старое поведение).
    
        Args:
            frame: Изображение в формате BGR.
            timestamp: Временная метка кадра.
            frame_idx: Индекс кадра (для логирования).
            total_frames: Общее количество кадров.
            parameter_table: Пользовательская таблица параметров.
    
        Returns:
            Результат анализа с timestamp.
        """
        frame_vlm_start = time.perf_counter()
    
        try:
            logger.info(
                "Frame %d/%d (%s): starting VLM analysis",
                frame_idx + 1, total_frames, timestamp,
            )

            # Формируем промпты с передачей таблицы параметров
            prompt_build_start = time.perf_counter()
            system_prompt, user_prompt = build_messages(
                frame_timestamp=timestamp,
                additional_context="Автоматический анализ мнемосхемы SCADA",
                parameter_table=parameter_table,
            )
            prompt_build_ms = (time.perf_counter() - prompt_build_start) * 1000
            logger.debug(
                "Frame %d/%d: prompt_build=%.1fms, system_len=%d, user_len=%d, has_table=%s",
                frame_idx + 1, total_frames, prompt_build_ms,
                len(system_prompt), len(user_prompt), parameter_table is not None,
            )
    
            # Отправляем запрос к VLM
            logger.info(
                "Frame %d/%d (%s): calling vlm_client.analyze_frame...",
                frame_idx + 1, total_frames, timestamp,
            )
            result = await self.vlm_client.analyze_frame(
                frame=frame,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
    
            # ВАЖНО: Проверка что result — это dict, а не list
            if isinstance(result, list):
                logger.warning(
                    "Frame %d/%d (%s): VLM returned list instead of dict — wrapping",
                    frame_idx + 1, total_frames, timestamp,
                )
                result = {"parameters": result}
            elif not isinstance(result, dict):
                logger.error(
                    "Frame %d/%d (%s): VLM returned unexpected type: %s — creating error result",
                    frame_idx + 1, total_frames, timestamp, type(result).__name__,
                )
                return {
                    "timestamp": timestamp,
                    "error": f"Unexpected VLM response type: {type(result).__name__}",
                    "parameters": [],
                    "frame_quality": "error",
                }

            # КРИТИЧНО: Валидация JSON через Pydantic перед использованием
            try:
                vlm_response = VLMResponse.model_validate(result)
                # Если валидация прошла — используем валидированные данные
                result["parameters"] = [p.model_dump() for p in vlm_response.parameters]
                result["vlm_validation"] = "success"
                
                logger.info(
                    "Frame %d/%d (%s): Pydantic validation PASSED, params=%d",
                    frame_idx + 1, total_frames, timestamp, vlm_response.param_count,
                )
            except ValidationError as e:
                logger.error(
                    "Frame %d/%d (%s): Pydantic validation FAILED — %s",
                    frame_idx + 1, total_frames, timestamp, str(e)[:500],
                )
                # НЕ генерируем XML из невалидных данных!
                return {
                    "timestamp": timestamp,
                    "error": f"VLM JSON validation failed: {str(e)[:200]}",
                    "parameters": [],
                    "frame_quality": "error",
                    "vlm_validation": "failed",
                }
    
            frame_vlm_elapsed_ms = (time.perf_counter() - frame_vlm_start) * 1000
            result["timestamp"] = timestamp
    
            params_count = len(result.get("parameters", []))
            json_parse_success = "error" not in result
    
            logger.info(
                "Frame %d/%d (%s): COMPLETE total=%.1fms, json_parse=%s, params_extracted=%d",
                frame_idx + 1, total_frames, timestamp,
                frame_vlm_elapsed_ms,
                "success" if json_parse_success else "failed",
                params_count,
            )
    
            return result
    
        except Exception as e:
            frame_vlm_elapsed_ms = (time.perf_counter() - frame_vlm_start) * 1000
            logger.error(
                "Frame %d/%d (%s): ERROR after %.1fms — %s: %s",
                frame_idx + 1, total_frames, timestamp,
                frame_vlm_elapsed_ms, type(e).__name__, str(e)[:500],
            )
            return {
                "timestamp": timestamp,
                "error": str(e),
                "parameters": [],
                "frame_quality": "error",
            }
    
    async def analyze_frames(
        self,
        frames: list[tuple[np.ndarray, str]],
        video_id: str = "",
        progress_callback: ProgressCallback | None = None,
        parameter_table: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Анализирует кадры через VLM с параллельной обработкой и пропуском похожих.
    
        Использует asyncio.Semaphore для контроля параллелизма.
        Пропускает кадры, похожие на предыдущий (если включено).
    
        Args:
            frames: Список кортежей (frame, timestamp).
            video_id: Идентификатор видео для логирования.
            progress_callback: Функция для отчётов о прогрессе.
            parameter_table: Пользовательская таблица параметров.
    
        Returns:
            Список результатов анализа с полями:
            - timestamp: временная метка кадра
            - parameters: список извлечённых параметров
            - mnemonic_id: идентификатор мнемосхемы (если найден)
            - frame_quality: оценка качества кадра
            - error: сообщение об ошибке (если есть)
            - skipped: True если кадр был пропущен (результат скопирован)
        """
        logger.info(
            "analyze_frames: total_frames=%d, skip_similar=%s, similarity_thresh=%.2f, concurrency=%d, has_table=%s",
            len(frames), self.skip_similar_frames, self.similarity_threshold,
            self.concurrency, parameter_table is not None,
        )

        results: list[dict[str, Any]] = [None] * len(frames)  # Предварительно заполняем None
        total = len(frames)
        semaphore = asyncio.Semaphore(self.concurrency)
        completed_count = 0
        lock = asyncio.Lock()
        skipped_count = 0
        processed_count = 0
        error_count = 0
    
        async def process_frame(
            idx: int,
            frame: np.ndarray,
            timestamp: str,
        ) -> tuple[int, dict[str, Any]]:
            """Обрабатывает один кадр с контролем семафора.
    
            Args:
                idx: Индекс кадра в списке.
                frame: Изображение в формате BGR.
                timestamp: Временная метка.
    
            Returns:
                Кортеж (idx, result) для сохранения порядка.
            """
            nonlocal completed_count, skipped_count, processed_count, error_count
    
            # Проверка схожести с предыдущим кадром
            if self.skip_similar_frames and self._prev_result is not None:
                similarity = self._compute_frame_similarity(frame)
                if similarity >= self.similarity_threshold:
                    logger.info(
                        "Frame %d/%d (%s): SKIPPED (similarity=%.2f with previous)",
                        idx + 1, total, timestamp, similarity,
                    )
                    # Копируем результат предыдущего кадра
                    result = dict(self._prev_result)
                    result["timestamp"] = timestamp
                    result["skipped"] = True
    
                    async with lock:
                        completed_count += 1
                        skipped_count += 1
                        if progress_callback:
                            progress_pct = 5.0 + (completed_count / total) * 80.0
                            progress_callback(video_id, progress_pct, "vlm_analysis", completed_count, total)
    
                    return idx, result
    
            # Анализ через VLM с семафором
            async with semaphore:
                result = await self._analyze_single_frame(
                    frame, timestamp, idx, total, parameter_table=parameter_table,
                )

                # Сохраняем для возможного пропуска следующего кадра
                if self.skip_similar_frames and "error" not in result:
                    self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self._prev_result = result
    
                result["skipped"] = False
    
                async with lock:
                    completed_count += 1
                    if "error" in result:
                        error_count += 1
                    else:
                        processed_count += 1
                    if progress_callback:
                        progress_pct = 5.0 + (completed_count / total) * 80.0
                        progress_callback(video_id, progress_pct, "vlm_analysis", completed_count, total)
    
                return idx, result
    
        # Запускаем все задачи параллельно
        logger.info("analyze_frames: launching %d frame tasks with semaphore(size=%d)", total, self.concurrency)
        launch_start = time.perf_counter()
        tasks = [
            process_frame(i, frame, timestamp)
            for i, (frame, timestamp) in enumerate(frames)
        ]
    
        # Собираем результаты с сохранением порядка
        logger.info("analyze_frames: waiting for all tasks to complete...")
        completed_results = await asyncio.gather(*tasks)
        launch_elapsed = (time.perf_counter() - launch_start) * 1000

        for idx, result in completed_results:
            results[idx] = result
    
        logger.info(
            "analyze_frames: COMPLETE in %.1fms — processed=%d, skipped=%d, errors=%d, total=%d",
            launch_elapsed, processed_count, skipped_count, error_count, total,
        )

        return results

    def normalize_parameters(self, raw_results: list[dict]) -> list[dict]:
        """Нормализует параметры: приведение типов, валидация диапазонов.

        Args:
            raw_results: Сырые результаты от VLM.

        Returns:
            Нормализованные результаты с валидными значениями.
        """
        normalized: list[dict] = []
        total_params = 0
        out_of_range_count = 0

        for result in raw_results:
            if "error" in result:
                normalized.append(result)
                continue

            params = result.get("parameters", [])
            normalized_params: list[dict] = []

            for param in params:
                try:
                    # КРИТИЧНО: Fallback для 'label' <- 'name' (VLM иногда возвращает 'name' вместо 'label')
                    if param.get("name") and not param.get("label"):
                        logger.warning(
                            "VLM returned 'name' instead of 'label' — normalizing: name='%s'",
                            param.get("name"),
                        )
                        param["label"] = param.pop("name")
                    
                    # Нормализация значения
                    value = param.get("value", "")
                    normalized_value = self._normalize_value(value)

                    # Определение типа параметра
                    param_type_str = param.get("param_type", "R")
                    try:
                        param_type = ParameterType(param_type_str)
                    except ValueError:
                        param_type = ParameterType.RESERVE

                    # Валидация диапазона
                    in_range = param.get("in_range", True)
                    if normalized_value != "" and param_type != ParameterType.RESERVE:
                        try:
                            numeric_value = float(normalized_value)
                            is_valid, _ = validate_value_in_range(numeric_value, param_type)
                            param["in_range"] = is_valid
                            if not is_valid:
                                out_of_range_count += 1
                        except ValueError:
                            param["in_range"] = False
                            out_of_range_count += 1

                    param["value"] = normalized_value
                    param["param_type"] = param_type.value
                    param["confidence"] = min(1.0, max(0.0, float(param.get("confidence", 0.5))))
                    normalized_params.append(param)
                    total_params += 1

                except Exception as e:
                    logger.warning("Ошибка нормализации параметра %s: %s", param, e)

            result["parameters"] = normalized_params
            normalized.append(result)

        # INFO: Normalization summary
        logger.info(
            "Normalization complete: total_params=%d, out_of_range=%d",
            total_params,
            out_of_range_count,
        )

        return normalized

    def temporal_smoothing(self, results: list[dict]) -> list[dict]:
        """Сглаживание: сравнивает последовательные кадры, отмечает аномалии.

        Анализирует изменения значений между соседними кадрами.
        Резкие изменения (>50% за один интервал 500мс) помечаются как аномалии.

        Args:
            results: Нормализованные результаты.

        Returns:
            Результаты с дополнительными флагами аномалий.
        """
        if len(results) < 2:
            return results

        # Кэш предыдущих значений по меткам параметров
        prev_values: dict[str, str] = {}
        total_anomalies = 0
        frame_anomaly_counts: dict[str, int] = {}  # timestamp -> anomaly count

        for result in results:
            if "error" in result:
                continue

            params = result.get("parameters", [])
            frame_anomalies = 0
            timestamp = result.get("timestamp", "?")

            for param in params:
                # КРИТИЧНО: Fallback для 'label' <- 'name' (VLM иногда возвращает 'name' вместо 'label')
                label = param.get("label") or param.get("name") or ""
                value = param.get("value", "")

                if not label or not value:
                    continue

                # Нормализуем: всегда используем 'label', удаляем 'name' если есть
                if "name" in param and "label" not in param:
                    param["label"] = param.pop("name")

                # Проверка на аномалию
                if label in prev_values:
                    prev_val = prev_values[label]
                    try:
                        prev_num = float(prev_val)
                        curr_num = float(value)

                        if prev_num != 0:
                            change_pct = abs((curr_num - prev_num) / prev_num) * 100
                            # Аномалия: изменение >50% за 500мс
                            if change_pct > 50:
                                param["anomaly"] = True
                                param["change_pct"] = round(change_pct, 1)
                                frame_anomalies += 1
                                total_anomalies += 1
                                # DEBUG: Individual anomaly detection
                                logger.debug(
                                    "Anomaly detected: label='%s', prev=%s, curr=%s, change=%.1f%%, frame_ts=%s",
                                    label,
                                    prev_val,
                                    value,
                                    change_pct,
                                    timestamp,
                                )
                                logger.debug(
                                    "Аномалия: %s изменился с %s на %s (%.1f%%)",
                                    label,
                                    prev_val,
                                    value,
                                    change_pct,
                                )

                    except ValueError:
                        pass  # Нечисловые значения не проверяем

                prev_values[label] = value

            if frame_anomalies > 0:
                frame_anomaly_counts[timestamp] = frame_anomalies

        # DEBUG/WARNING: Temporal smoothing summary
        if total_anomalies > 0:
            logger.warning(
                "Temporal smoothing: total_anomalies=%d across %d frames",
                total_anomalies,
                len(frame_anomaly_counts),
            )
            for ts, count in frame_anomaly_counts.items():
                logger.debug("  Frame %s: %d anomalies", ts, count)
        else:
            logger.debug("Temporal smoothing: no anomalies detected")

        return results

    def _normalize_parameter_labels(
        self,
        results: list[dict],
        parameter_table: list[dict] | None = None,
        threshold: float = 0.80,
    ) -> list[dict]:
        """Нормализует лейблы VLM через fuzzy matching с таблицей параметров.

        КРИТИЧНО: VLM возвращает то, что ВИДИТ на экране (например "Т газа"),
        а Python маппит на полное название из таблицы ("Температура газа").

        УЛУЧШЕНО: Удаляет теги типа (PT4413) перед сравнением.

        Args:
            results: Результаты анализа VLM.
            parameter_table: Таблица параметров (id, name, unit, type, ...).
            threshold: Порог схожести (0.80 = 80%).

        Returns:
            Результаты с нормализованными лейблами и ID из таблицы.
        """
        if not parameter_table:
            logger.warning("No parameter table — skipping label normalization")
            return results

        # Строим маппинг имя->ID из таблицы
        table_name_to_id: dict[str, int] = {p['name']: p['id'] for p in parameter_table}
        table_names = list(table_name_to_id.keys())
        
        logger.info(
            "Label normalization: %d parameters in table, threshold=%.0f%%",
            len(table_names),
            threshold * 100,
        )

        normalized_count = 0
        skipped_count = 0

        for result in results:
            params = result.get("parameters", [])
            
            for param in params:
                # КРИТИЧНО: Fallback для 'label' <- 'name'
                raw_label = param.get("label") or param.get("name") or ""
                if not raw_label:
                    continue

                # Ищем лучшее совпадение
                best_match = None
                best_score = 0.0

                for table_name in table_names:
                    # УДАЛЯЕМ ТЕГИ типа (PT4413) перед сравнением
                    clean_vlm = re.sub(r'\([^)]*\)', '', raw_label).strip()
                    clean_table = re.sub(r'\([^)]*\)', '', table_name).strip()
                    
                    # SequenceMatcher: 0.0 = нет совпадений, 1.0 = точное совпадение
                    score = SequenceMatcher(None, clean_vlm.lower(), clean_table.lower()).ratio()
                    
                    if score > best_score:
                        best_score = score
                        best_match = table_name

                # Проверяем порог
                if best_match and best_score >= threshold:
                    # Нашли совпадение > 80% — используем имя из таблицы
                    param["label"] = best_match
                    param["label_confidence"] = best_score
                    normalized_count += 1
                    
                    if best_score < 1.0:
                        logger.debug(
                            "Label normalized: '%s' → '%s' (similarity=%.0f%%)",
                            raw_label, best_match, best_score * 100,
                        )
                else:
                    # Не нашли совпадение — помечаем для пропуска
                    param["label"] = raw_label
                    param["label_confidence"] = best_score
                    param["skip"] = True  # Пометить для пропуска в XML
                    skipped_count += 1
                    
                    logger.warning(
                        "Label not matched: '%s' (best='%s', similarity=%.0f%%) — will skip",
                        raw_label, best_match or "none", best_score * 100,
                    )

        logger.info(
            "Label normalization complete: normalized=%d, skipped=%d",
            normalized_count, skipped_count,
        )

        return results

    def _generate_xml_output(
        self,
        results: list[dict],
        video_id: str,
        parameter_table: list[dict] | None = None,
    ) -> tuple[str, Path]:
        """Генерирует XML из результатов анализа с 500мс снапшотами.

        Если VLM анализирует кадры каждые 1000мс, дублирует результаты
        для промежуточных 500мс снапшотов, чтобы XML соответствовал спецификации.
        
        ИСПОЛЬЗУЕТ ID ИЗ ТАБЛИЦЫ ПАРАМЕТРОВ вместо динамической генерации.

        Args:
            results: Результаты анализа VLM.
            video_id: Идентификатор видео.

        Returns:
            Кортеж (xml_content, xml_path).
        """
        # Создаём snapshots из результатов
        snapshots = []
        
        # КРИТИЧНО: Строим маппинг имя->ID из таблицы параметров
        # Это гарантирует стабильные ID across всех кадров!
        table_name_to_id: dict[str, int] = {}
        if parameter_table:
            table_name_to_id = {p['name']: p['id'] for p in parameter_table}
            logger.info(
                "XML generation: loaded %d parameter IDs from table",
                len(table_name_to_id),
            )
        
        # Fallback: динамический счётчик для параметров НЕ из таблицы
        param_counter = max(table_name_to_id.values(), default=0) + 1
        param_id_map: dict[str, int] = {}  # label -> param_id (для новых параметров)

        for result in results:
            if "error" in result:
                continue

            timestamp = result.get("timestamp", "00:00:00.000")
            params = result.get("parameters", [])

            param_values: dict[int, str] = {}
            param_metadata: dict[int, tuple[str, str, str]] = {}

            for param in params:
                # КРИТИЧНО: Пропускаем параметры, которые не прошли fuzzy matching
                if param.get("skip"):
                    continue
                
                # КРИТИЧНО: Fallback для 'label' <- 'name'
                label = param.get("label") or param.get("name") or ""
                value = param.get("value", "")
                unit = param.get("unit", "")
                param_type = param.get("param_type", "R")

                if not label or not value:
                    continue

                # Нормализуем: всегда используем 'label'
                if "name" in param and "label" not in param:
                    param["label"] = param.pop("name")

                # ПОЛУЧАЕМ ID ИЗ ТАБЛИЦЫ (не генерируем динамически!)
                pid = table_name_to_id.get(label)
                
                if pid is None:
                    # Параметр НЕ в таблице — возможно галлюцинация или новый параметр
                    # Используем fallback: назначаем новый ID
                    if label not in param_id_map:
                        param_id_map[label] = param_counter
                        param_counter += 1
                        logger.warning(
                            "Parameter '%s' not in table — assigned dynamic ID=%d",
                            label, pid,
                        )
                    pid = param_id_map[label]

                param_values[pid] = value
                param_metadata[pid] = (param_type, label, unit)

            if param_values:
                # Создаём снапшот для текущего кадра
                snapshot = create_snapshot(
                    timestamp=timestamp,
                    param_values=param_values,
                    param_metadata=param_metadata,
                )
                snapshots.append(snapshot)

                # Если frame_interval_ms > 500, создаём промежуточный снапшот
                # (дублируем результаты для 500мс интервала XML)
                if self.frame_interval_ms >= 1000:
                    intermediate_ts = self._get_intermediate_timestamp(timestamp)
                    intermediate_snapshot = create_snapshot(
                        timestamp=intermediate_ts,
                        param_values=param_values,
                        param_metadata=param_metadata,
                    )
                    snapshots.append(intermediate_snapshot)

        # Генерируем XML
        xml_content = generate_xml(snapshots, scheme_id=video_id)

        # Сохраняем в файл
        output_dir = Path(self.settings.output_xml_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        xml_path = output_dir / f"{video_id}_output.xml"
        xml_path.write_text(xml_content, encoding="utf-8")

        # INFO: XML generation summary
        total_params_in_xml = len(param_id_map)
        logger.info(
            "XML generation: output_path=%s, snapshots=%d (with 500ms interpolation), total_unique_params=%d",
            xml_path,
            len(snapshots),
            total_params_in_xml,
        )
        logger.info("XML сохранён: %s (%d snapshots)", xml_path, len(snapshots))

        return xml_content, xml_path

    def _get_intermediate_timestamp(self, timestamp: str) -> str:
        """Вычисляет промежуточный timestamp (на 500мс позже).

        Для timestamp "00:00:00.001" возвращает "00:00:00.501".
        Для timestamp "00:00:01.001" возвращает "00:00:01.501".

        Args:
            timestamp: Исходный timestamp в формате HH:MM:SS.mmm.

        Returns:
            Timestamp на 500мс позже.
        """
        # Парсим timestamp
        parts = timestamp.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        sec_msec = parts[2].split(".")
        seconds = int(sec_msec[0])
        millis = int(sec_msec[1]) if len(sec_msec) > 1 else 0

        # Добавляем 500мс
        total_ms = hours * 3_600_000 + minutes * 60_000 + seconds * 1000 + millis
        total_ms += 500

        # Форматируем обратно
        return self._format_timestamp_ms(total_ms)

    def _format_timestamp_ms(self, ms: int) -> str:
        """Форматирует миллисекунды в строку HH:MM:SS.mmm.

        Args:
            ms: Время в миллисекундах.

        Returns:
            Строка в формате HH:MM:SS.mmm.
        """
        hours = ms // 3_600_000
        minutes = (ms % 3_600_000) // 60_000
        seconds = (ms % 60_000) // 1000
        millis = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    def _normalize_value(self, value: str) -> str:
        """Нормализует значение параметра.

        - Заменяет запятую на точку
        - Удаляет лишние пробелы
        - Убирает единицы измерения из значения

        Args:
            value: Сырое значение.

        Returns:
            Нормализованное значение.
        """
        if not value:
            return ""

        # Убираем пробелы
        normalized = value.strip()

        # Заменяем запятую на точку
        normalized = normalized.replace(",", ".")

        # Удаляем常见 единицы измерения из конца значения
        units_to_remove = ["°C", "°С", "C", "С", "кПа", "кPa", "кпа", "kPa", "мм/с", "мм", "mm", "Гц", "Hz", "%", "В", "V"]
        for unit in units_to_remove:
            if normalized.endswith(unit):
                normalized = normalized[:-len(unit)].strip()
                break

        return normalized

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Применяет CLAHE для улучшения контраста.

        Args:
            frame: Входной кадр в BGR формате.

        Returns:
            Кадр с улучшенным контрастом.
        """
        # Конвертируем в LAB цветовое пространство
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Применяем CLAHE к L каналу
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Конвертируем обратно в BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    async def close(self) -> None:
        """Закрывает ресурсы пайплайна."""
        # VLMClientDirect — singleton, не закрываем при завершении пайплайна
        try:
            from app.core.vlm_client_direct import VLMClientDirect
            if isinstance(self.vlm_client, VLMClientDirect):
                logger.info("VLMPipeline закрыт (VLMClientDirect singleton сохранён)")
                return
        except ImportError:
            pass

        # Обычный HTTP VLMClient — закрываем соединение
        if hasattr(self.vlm_client, "close"):
            await self.vlm_client.close()
        logger.info("VLMPipeline закрыт")

    async def __aenter__(self) -> "VLMPipeline":
        """Асинхронный контекстный менеджер - вход."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Асинхронный контекстный менеджер - выход."""
        await self.close()


async def process_video_with_vlm(
    video_path: Path,
    video_id: str,
    send_email: bool = True,
    progress_callback: ProgressCallback | None = None,
    parameter_table: list[dict] | None = None,
) -> PipelineResult:
    """Удобная функция для обработки видео через VLM.

    Args:
        video_path: Путь к видеофайлу.
        video_id: Идентификатор видео.
        send_email: Отправить зашифрованный XML по email.
        progress_callback: Функция для отчётов о прогрессе.
        parameter_table: Пользовательская таблица параметров из xlsx/csv.

    Returns:
        PipelineResult с результатами обработки.

    Example:
        >>> from pathlib import Path
        >>> result = await process_video_with_vlm(
        ...     Path("video.mp4"),
        ...     "test-video-id",
        ...     send_email=False,
        ... )
        >>> print(result.status)
        'completed'
    """
    async with VLMPipeline() as pipeline:
        return await pipeline.process_video(
            video_path=video_path,
            video_id=video_id,
            send_email=send_email,
            progress_callback=progress_callback,
            parameter_table=parameter_table,
        )
