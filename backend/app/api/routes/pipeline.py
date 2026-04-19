"""API-роутер для управления VLM-конвейером обработки видео.

Модуль предоставляет REST API для:
- Запуска обработки видео через VLM (Qwen3.5-4B)
- Мониторинга статуса и прогресса
- Получения результатов XML
- Шифрования и отправки email
- Загрузки пользовательских таблиц параметров (xlsx/csv)
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api.websocket import ws_manager
from app.config import settings
from app.core.vlm_client import get_vlm_client

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic модели для API
# ---------------------------------------------------------------------------


class PipelineStartResponse(BaseModel):
    """Ответ при запуске конвейера."""

    status: str
    video_id: str
    video_path: str
    message: str = ""


class PipelineStatusResponse(BaseModel):
    """Ответ со статусом конвейера."""

    video_id: str
    status: str  # idle | starting | processing | completed | failed | partial
    progress_pct: float = 0.0
    current_step: str = ""
    frames_processed: int = 0
    total_frames: int = 0
    parameters_extracted: int = 0
    processing_time_seconds: float = 0.0
    xml_path: str = ""
    encrypted_path: str = ""
    email_sent: bool = False
    errors: list[str] = Field(default_factory=list)


class FrameInfo(BaseModel):
    """Информация об извлечённом кадре."""

    index: int
    timestamp: str
    path: str = ""  # Путь к сохранённому кадру (если есть)


class FramesListResponse(BaseModel):
    """Ответ со списком кадров."""

    video_id: str
    total_frames: int
    interval_ms: int
    frames: list[FrameInfo]


class XmlResponse(BaseModel):
    """Ответ с XML-контентом."""

    video_id: str
    xml: str
    xml_path: str


class EmailSendResponse(BaseModel):
    """Ответ при отправке email."""

    video_id: str
    status: str  # sent | error
    message: str = ""


class VlmHealthResponse(BaseModel):
    """Ответ о состоянии VLM-сервера."""

    healthy: bool
    model_loaded: bool = True
    server_url: str
    message: str = ""


class ParameterTableUploadResponse(BaseModel):
    """Ответ при загрузке таблицы параметров."""

    status: str
    parameter_count: int
    message: str = ""


class ParameterTableEntry(BaseModel):
    """Одна запись из таблицы параметров."""

    id: int
    name: str
    unit: str = ""
    short_name: str = ""
    decimal_places: int = 1
    sheet_name: str = ""


# ---------------------------------------------------------------------------
# Глобальное состояние конвейера
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    """Состояние конвейера для отслеживания прогресса."""

    video_id: str = ""
    status: str = "idle"
    progress_pct: float = 0.0
    current_step: str = ""
    frames_processed: int = 0
    total_frames: int = 0
    parameters_extracted: int = 0
    processing_time_seconds: float = 0.0
    xml_path: str = ""
    encrypted_path: str = ""
    email_sent: bool = False
    errors: list[str] = field(default_factory=list)

    # Хранение результатов по video_id
    results: dict[str, dict[str, Any]] = field(default_factory=dict)


# Глобальное состояние с блокировкой для потокобезопасности
_pipeline_state = PipelineState()
_state_lock = threading.Lock()

# Хранение запущенных задач
_active_tasks: dict[str, asyncio.Task] = {}

# Хранение загруженных таблиц параметров по video_id
_parameter_tables: dict[str, list[dict]] = {}
_parameter_table_lock = threading.Lock()


def _get_state_copy() -> dict[str, Any]:
    """Возвращает копию текущего состояния (потокобезопасно)."""
    with _state_lock:
        return {
            "video_id": _pipeline_state.video_id,
            "status": _pipeline_state.status,
            "progress_pct": _pipeline_state.progress_pct,
            "current_step": _pipeline_state.current_step,
            "frames_processed": _pipeline_state.frames_processed,
            "total_frames": _pipeline_state.total_frames,
            "parameters_extracted": _pipeline_state.parameters_extracted,
            "processing_time_seconds": _pipeline_state.processing_time_seconds,
            "xml_path": _pipeline_state.xml_path,
            "encrypted_path": _pipeline_state.encrypted_path,
            "email_sent": _pipeline_state.email_sent,
            "errors": _pipeline_state.errors.copy(),
        }


def _update_state(**kwargs) -> None:
    """Обновляет состояние конвейера (потокобезопасно)."""
    with _state_lock:
        for key, value in kwargs.items():
            if hasattr(_pipeline_state, key):
                setattr(_pipeline_state, key, value)
                # INFO: Pipeline state changes
                if key == "status":
                    logger.info(
                        "Pipeline state change: video_id=%s, field=status, value=%s",
                        _pipeline_state.video_id,
                        value,
                    )


def _progress_callback(
    video_id: str,
    progress_pct: float,
    current_step: str,
    frames_processed: int,
    total_frames: int,
) -> None:
    """Callback для отчётов о прогрессе из VLMPipeline.

    Args:
        video_id: ID обрабатываемого видео.
        progress_pct: Процент выполнения (0-100).
        current_step: Текущий этап обработки.
        frames_processed: Количество обработанных кадров.
        total_frames: Общее количество кадров.
    """
    # Определяем статус на основе прогресса
    status = "processing"
    if progress_pct >= 100.0:
        status = "completed"
    elif current_step == "starting":
        status = "starting"

    _update_state(
        progress_pct=progress_pct,
        current_step=current_step,
        frames_processed=frames_processed,
        total_frames=total_frames,
        status=status,
    )

    # Отправляем WebSocket-обновление
    try:
        # Python 3.10+: get_running_loop() is the correct way to get the current event loop
        loop = asyncio.get_running_loop()
        # DEBUG: WebSocket broadcast
        logger.debug(
            "WebSocket broadcast: type=progress, video_id=%s, progress_pct=%.1f%%, step=%s, frames=%d/%d",
            video_id,
            progress_pct,
            current_step,
            frames_processed,
            total_frames,
        )
        loop.create_task(
            ws_manager.broadcast({
                "type": "progress",
                "video_id": video_id,
                "progress_pct": progress_pct,
                "current_step": current_step,
                "frames_processed": frames_processed,
                "total_frames": total_frames,
                "status": status,
            })
        )
    except RuntimeError:
        # Нет running event loop — логируем для отладки
        logger.warning(
            "Cannot broadcast progress: no running event loop (video_id=%s, step=%s)",
            video_id,
            current_step,
        )


async def _run_vlm_pipeline(video_path: Path, video_id: str) -> None:
    """Фоновая задача: обработка видео через VLM.

    Args:
        video_path: Путь к видеофайлу.
        video_id: Идентификатор видео.
    """
    from app.core.vlm_pipeline import VLMPipeline

    _update_state(
        video_id=video_id,
        status="starting",
        progress_pct=0.0,
        current_step="initializing",
        frames_processed=0,
        total_frames=0,
        parameters_extracted=0,
        errors=[],
    )

    # INFO: Pipeline state change at start
    logger.info(
        "Pipeline state change: video_id=%s, status=starting, step=initializing",
        video_id,
    )

    # WebSocket: начало обработки
    logger.debug(
        "WebSocket broadcast: type=pipeline_started, video_id=%s, video_path=%s",
        video_id,
        video_path,
    )
    await ws_manager.broadcast({
        "type": "pipeline_started",
        "video_id": video_id,
        "video_path": str(video_path),
        "status": "starting",
    })

    try:
        # Получаем таблицу параметров если загружена
        param_table = None
        with _parameter_table_lock:
            param_table = _parameter_tables.get(video_id) or _parameter_tables.get("__global__")

        if param_table:
            logger.info(
                "Using parameter table for video %s: %d params",
                video_id,
                len(param_table),
            )

        async with VLMPipeline() as pipeline:
            result = await pipeline.process_video(
                video_path=video_path,
                video_id=video_id,
                send_email=False,  # Email отправляется отдельно по кнопке
                progress_callback=_progress_callback,
                parameter_table=param_table,
            )

            # Обновляем состояние по результатам
            _update_state(
                status=result.status,
                progress_pct=100.0,
                current_step="completed",
                frames_processed=result.processed_frames,
                total_frames=result.total_frames,
                parameters_extracted=result.total_parameters,
                processing_time_seconds=result.processing_time_seconds,
                xml_path=str(result.xml_path) if result.xml_path else "",
                encrypted_path=str(result.encrypted_path) if result.encrypted_path else "",
                errors=result.errors,
            )

            # Сохраняем результат в кэш
            with _state_lock:
                _pipeline_state.results[video_id] = {
                    "status": result.status,
                    "xml_path": str(result.xml_path) if result.xml_path else "",
                    "encrypted_path": str(result.encrypted_path) if result.encrypted_path else "",
                    "total_frames": result.total_frames,
                    "processed_frames": result.processed_frames,
                    "total_parameters": result.total_parameters,
                    "processing_time_seconds": result.processing_time_seconds,
                    "errors": result.errors,
                }

            # WebSocket: завершение обработки
            logger.debug(
                "WebSocket broadcast: type=pipeline_complete, video_id=%s, status=%s, frames=%d/%d, params=%d",
                video_id,
                result.status,
                result.processed_frames,
                result.total_frames,
                result.total_parameters,
            )
            await ws_manager.broadcast({
                "type": "pipeline_complete",
                "video_id": video_id,
                "status": result.status,
                "total_frames": result.total_frames,
                "processed_frames": result.processed_frames,
                "total_parameters": result.total_parameters,
                "processing_time_seconds": result.processing_time_seconds,
                "xml_path": str(result.xml_path) if result.xml_path else "",
            })

            logger.info(
                "VLM Pipeline завершён: video_id=%s, status=%s, frames=%d/%d, params=%d",
                video_id,
                result.status,
                result.processed_frames,
                result.total_frames,
                result.total_parameters,
            )

    except Exception as e:
        # ERROR: Pipeline failure
        logger.exception("Ошибка VLM Pipeline: %s", e)
        logger.error(
            "Pipeline error: video_id=%s, error_type=%s, error_msg=%s",
            video_id,
            type(e).__name__,
            str(e)[:200],
        )
        _update_state(
            status="failed",
            current_step="error",
            errors=[str(e)],
        )

        # WebSocket: ошибка
        logger.debug(
            "WebSocket broadcast: type=pipeline_error, video_id=%s, error=%s",
            video_id,
            str(e)[:100],
        )
        await ws_manager.broadcast({
            "type": "pipeline_error",
            "video_id": video_id,
            "message": str(e),
        })

    finally:
        # Удаляем задачу из активных
        if video_id in _active_tasks:
            del _active_tasks[video_id]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@router.post("/start/{video_id}", response_model=PipelineStartResponse)
async def start_pipeline(video_id: str) -> PipelineStartResponse:
    """Запускает VLM-конвейер обработки видео.

    Args:
        video_id: ID загруженного видео.

    Returns:
        Информация о запущенной задаче.

    Raises:
        HTTPException: Если видео не найдено или уже обрабатывается.
    """
    # INFO: Endpoint hit
    logger.info(
        "Endpoint hit: method=POST, path=/start/%s, video_id=%s",
        video_id,
        video_id,
    )

    # Ищем видеофайл
    input_dir = Path(settings.input_videos_dir)
    files = list(input_dir.glob(f"{video_id}_*"))

    if not files:
        # WARNING: Video not found
        logger.warning(
            "Error response: status_code=404, error_detail='Видео не найдено', video_id=%s",
            video_id,
        )
        raise HTTPException(status_code=404, detail="Видео не найдено")

    video_path = files[0]

    # Проверяем, не обрабатывается ли уже это видео
    if video_id in _active_tasks and not _active_tasks[video_id].done():
        # WARNING: Conflict - video already processing
        logger.warning(
            "Error response: status_code=409, error_detail='Видео уже обрабатывается', video_id=%s",
            video_id,
        )
        raise HTTPException(
            status_code=409,
            detail="Видео уже обрабатывается",
        )

    # Запускаем фоновую задачу
    task = asyncio.create_task(_run_vlm_pipeline(video_path, video_id))
    _active_tasks[video_id] = task

    logger.info("Запущена обработка видео: %s (ID: %s)", video_path, video_id)

    return PipelineStartResponse(
        status="started",
        video_id=video_id,
        video_path=str(video_path),
        message="Обработка запущена в фоновом режиме",
    )


@router.get("/status", response_model=PipelineStatusResponse)
async def get_pipeline_status() -> PipelineStatusResponse:
    """Возвращает текущий статус конвейера.

    Returns:
        Текущее состояние конвейера.
    """
    # INFO: Endpoint hit
    logger.info(
        "Endpoint hit: method=GET, path=/status, current_video_id=%s, status=%s",
        _pipeline_state.video_id,
        _pipeline_state.status,
    )
    state = _get_state_copy()
    return PipelineStatusResponse(**state)


@router.get("/status/{video_id}", response_model=PipelineStatusResponse)
async def get_video_status(video_id: str) -> PipelineStatusResponse:
    """Возвращает статус обработки конкретного видео.

    Args:
        video_id: ID видео.

    Returns:
        Состояние обработки указанного видео.
    """
    # INFO: Endpoint hit
    logger.info(
        "Endpoint hit: method=GET, path=/status/%s, video_id=%s",
        video_id,
        video_id,
    )

    # Сначала проверяем кэш результатов
    with _state_lock:
        if video_id in _pipeline_state.results:
            cached = _pipeline_state.results[video_id]
            return PipelineStatusResponse(
                video_id=video_id,
                status=cached.get("status", "completed"),
                progress_pct=100.0,
                current_step="completed",
                frames_processed=cached.get("processed_frames", 0),
                total_frames=cached.get("total_frames", 0),
                parameters_extracted=cached.get("total_parameters", 0),
                processing_time_seconds=cached.get("processing_time_seconds", 0.0),
                xml_path=cached.get("xml_path", ""),
                encrypted_path=cached.get("encrypted_path", ""),
                errors=cached.get("errors", []),
            )

    # Если не в кэше, проверяем текущее состояние
    state = _get_state_copy()
    if state["video_id"] == video_id:
        return PipelineStatusResponse(**state)

    # Видео не найдено
    return PipelineStatusResponse(
        video_id=video_id,
        status="not_found",
        errors=["Видео не найдено в кэше результатов"],
    )


@router.get("/completed", response_model=list[PipelineStatusResponse])
async def get_completed_videos() -> list[PipelineStatusResponse]:
    """Возвращает список всех завершенных видео из кэша результатов.

    Returns:
        Список статусов всех обработанных видео.
    """
    # INFO: Endpoint hit
    logger.info("Endpoint hit: method=GET, path=/completed")

    with _state_lock:
        if not _pipeline_state.results:
            return []

        completed_videos = []
        for video_id, cached in _pipeline_state.results.items():
            status = cached.get("status", "completed")
            if status == "completed":
                completed_videos.append(PipelineStatusResponse(
                    video_id=video_id,
                    status=status,
                    progress_pct=100.0,
                    current_step="completed",
                    frames_processed=cached.get("processed_frames", 0),
                    total_frames=cached.get("total_frames", 0),
                    parameters_extracted=cached.get("total_parameters", 0),
                    processing_time_seconds=cached.get("processing_time_seconds", 0.0),
                    xml_path=cached.get("xml_path", ""),
                    encrypted_path=cached.get("encrypted_path", ""),
                    email_sent=cached.get("email_sent", False),
                    errors=cached.get("errors", []),
                ))

        return completed_videos


@router.get("/xml/{video_id}", response_model=XmlResponse)
async def get_xml_output(video_id: str) -> XmlResponse:
    """Возвращает сгенерированный XML для видео.

    Args:
        video_id: ID видео.

    Returns:
        XML-контент.

    Raises:
        HTTPException: Если XML не найден.
    """
    # INFO: Endpoint hit
    logger.info(
        "Endpoint hit: method=GET, path=/xml/%s, video_id=%s",
        video_id,
        video_id,
    )

    # Проверяем кэш результатов
    xml_path_str = ""
    with _state_lock:
        if video_id in _pipeline_state.results:
            xml_path_str = _pipeline_state.results[video_id].get("xml_path", "")

    # Если не в кэше, ищем файл
    if not xml_path_str:
        output_dir = Path(settings.output_xml_dir)
        xml_path = output_dir / f"{video_id}_output.xml"
        if xml_path.exists():
            xml_path_str = str(xml_path)

    if not xml_path_str:
        # WARNING: XML not found
        logger.warning(
            "Error response: status_code=404, error_detail='XML не найден для данного видео', video_id=%s",
            video_id,
        )
        raise HTTPException(
            status_code=404,
            detail="XML не найден для данного видео. Сначала запустите pipeline.",
        )

    xml_path = Path(xml_path_str)
    if not xml_path.exists():
        # WARNING: XML file not on disk
        logger.warning(
            "Error response: status_code=404, error_detail='XML файл не найден на диске', video_id=%s, path=%s",
            video_id,
            xml_path_str,
        )
        raise HTTPException(status_code=404, detail="XML файл не найден на диске")

    xml_content = xml_path.read_text(encoding="utf-8")

    return XmlResponse(
        video_id=video_id,
        xml=xml_content,
        xml_path=str(xml_path),
    )


@router.post("/send-email/{video_id}", response_model=EmailSendResponse)
async def send_email(video_id: str) -> EmailSendResponse:
    """Отправляет зашифрованный XML по email.

    Args:
        video_id: ID видео.

    Returns:
        Статус отправки.

    Raises:
        HTTPException: Если XML не найден или ошибка отправки.
    """
    # INFO: Endpoint hit
    logger.info(
        "Endpoint hit: method=POST, path=/send-email/%s, video_id=%s",
        video_id,
        video_id,
    )

    from app.core.crypto_service import encrypt_xml
    from app.core.email_service import send_xml_email

    # Ищем XML файл
    xml_path_str = ""
    with _state_lock:
        if video_id in _pipeline_state.results:
            xml_path_str = _pipeline_state.results[video_id].get("xml_path", "")

    if not xml_path_str:
        output_dir = Path(settings.output_xml_dir)
        xml_path = output_dir / f"{video_id}_output.xml"
        if xml_path.exists():
            xml_path_str = str(xml_path)

    if not xml_path_str:
        # WARNING: XML not found for email
        logger.warning(
            "Error response: status_code=404, error_detail='XML не найден для отправки email', video_id=%s",
            video_id,
        )
        raise HTTPException(
            status_code=404,
            detail="XML не найден. Сначала запустите pipeline.",
        )

    xml_path = Path(xml_path_str)
    if not xml_path.exists():
        # WARNING: XML file not on disk for email
        logger.warning(
            "Error response: status_code=404, error_detail='XML файл не найден на диске для email', video_id=%s",
            video_id,
        )
        raise HTTPException(status_code=404, detail="XML файл не найден на диске")

    xml_content = xml_path.read_text(encoding="utf-8")

    # Шифрование
    try:
        encrypted = encrypt_xml(xml_content)
        encrypted_path = xml_path.with_suffix(".xml.gpg")
        encrypted_path.write_bytes(encrypted)
        logger.info("XML зашифрован: %s", encrypted_path)
    except Exception as e:
        # ERROR: GPG encryption failure
        logger.error(
            "Error response: status_code=500, error_type=gpg_encryption, error_msg=%s, video_id=%s",
            str(e)[:100],
            video_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка GPG шифрования: {e}",
        )

    # Отправка email
    try:
        sent = send_xml_email(
            encrypted_data=encrypted,
            filename=f"{video_id}_output.xml.gpg",
            subject=f"InfoDiode: SCADA Data from {video_id}",
        )

        if sent:
            # Обновляем кэш
            with _state_lock:
                if video_id in _pipeline_state.results:
                    _pipeline_state.results[video_id]["email_sent"] = True

            logger.info("Email отправлен для видео %s", video_id)
            return EmailSendResponse(
                video_id=video_id,
                status="sent",
                message="Email успешно отправлен",
            )
        else:
            # ERROR: Email send failure
            logger.error(
                "Error response: status_code=500, error_type=email_send, error_msg='Не удалось отправить email', video_id=%s",
                video_id,
            )
            raise HTTPException(
                status_code=500,
                detail="Не удалось отправить email",
            )

    except HTTPException:
        raise
    except Exception as e:
        # ERROR: Email send exception
        logger.error(
            "Error response: status_code=500, error_type=email_exception, error_msg=%s, video_id=%s",
            str(e)[:100],
            video_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка отправки email: {e}",
        )


@router.get("/frames/{video_id}", response_model=FramesListResponse)
async def get_frames_list(video_id: str) -> FramesListResponse:
    """Возвращает список извлечённых кадров для видео.

    Args:
        video_id: ID видео.

    Returns:
        Список кадров с временными метками.

    Raises:
        HTTPException: Если видео не найдено.
    """
    # INFO: Endpoint hit
    logger.info(
        "Endpoint hit: method=GET, path=/frames/%s, video_id=%s",
        video_id,
        video_id,
    )

    import cv2

    # Ищем видеофайл
    input_dir = Path(settings.input_videos_dir)
    files = list(input_dir.glob(f"{video_id}_*"))

    if not files:
        # WARNING: Video not found for frames
        logger.warning(
            "Error response: status_code=404, error_detail='Видео не найдено для frames', video_id=%s",
            video_id,
        )
        raise HTTPException(status_code=404, detail="Видео не найдено")

    video_path = files[0]

    # Получаем информацию о видео
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        # WARNING: Cannot open video
        logger.warning(
            "Error response: status_code=400, error_detail='Не удалось открыть видео', video_id=%s, path=%s",
            video_id,
            video_path,
        )
        raise HTTPException(status_code=400, detail="Не удалось открыть видео")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int(total_frames / fps * 1000) if fps > 0 else 0

        interval_ms = settings.vlm_frame_interval_ms

        # Генерируем список временных меток
        frames: list[FrameInfo] = []
        current_ms = 1  # Начинаем с 1мс (получим .001)
        frame_idx = 0

        while current_ms < duration_ms:
            hours = current_ms // 3_600_000
            minutes = (current_ms % 3_600_000) // 60_000
            seconds = (current_ms % 60_000) // 1000
            millis = current_ms % 1000
            timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

            frames.append(FrameInfo(
                index=frame_idx,
                timestamp=timestamp,
            ))

            frame_idx += 1
            current_ms += interval_ms

        return FramesListResponse(
            video_id=video_id,
            total_frames=len(frames),
            interval_ms=interval_ms,
            frames=frames,
        )

    finally:
        cap.release()


@router.get("/vlm-health", response_model=VlmHealthResponse)
async def vlm_health_check() -> VlmHealthResponse:
    """Проверяет доступность VLM-сервера (llama-server).

    Returns:
        Статус доступности VLM.
    """
    # INFO: Endpoint hit
    logger.info(
        "Endpoint hit: method=GET, path=/vlm-health, server_url=%s",
        settings.vlm_base_url,
    )

    try:
        client = get_vlm_client()
        healthy = await client.health_check()
        # VLMClientDirect — singleton, не закрываем
        try:
            from app.core.vlm_client_direct import VLMClientDirect
            if not isinstance(client, VLMClientDirect):
                await client.close()
        except ImportError:
            await client.close()

        if healthy:
            return VlmHealthResponse(
                healthy=True,
                model_loaded=True,
                server_url=settings.vlm_base_url,
                message="VLM сервер доступен",
            )
        else:
            return VlmHealthResponse(
                healthy=False,
                model_loaded=False,
                server_url=settings.vlm_base_url,
                message="VLM сервер не отвечает",
            )

    except Exception as e:
        return VlmHealthResponse(
            healthy=False,
            model_loaded=False,
            server_url=settings.vlm_base_url,
            message=f"Ошибка проверки: {str(e)}",
        )


@router.post("/upload-parameter-table", response_model=ParameterTableUploadResponse)
async def upload_parameter_table(
    file: UploadFile,
    video_id: str = "",
) -> ParameterTableUploadResponse:
    """Загружает пользовательскую таблицу параметров (xlsx/csv).

    Таблица используется при VLM-анализе для точного сопоставления
    распознанных меток с ID параметров из спецификации.

    Args:
        file: Файл таблицы (.xlsx или .csv).
        video_id: ID видео, к которому привязана таблица (опционально).

    Returns:
        Информация о загруженной таблице.
    """
    from app.core.parameter_mapper import load_parameter_table
    import tempfile

    # Валидация расширения файла
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".xlsx", ".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат: {suffix}. Используйте .xlsx или .csv",
        )

    # Сохраняем во временный файл для парсинга
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=settings.input_videos_dir
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка сохранения файла: {e}",
        )

    # Парсим таблицу
    try:
        params = load_parameter_table(tmp_path)
    except ValueError as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка парсинга таблицы: {e}",
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if not params:
        raise HTTPException(
            status_code=400,
            detail="Таблица параметров пуста или не содержит распознаваемых колонок",
        )

    # Сохраняем в память по video_id или глобально
    key = video_id if video_id else "__global__"
    with _parameter_table_lock:
        _parameter_tables[key] = params

    logger.info(
        "Загружена таблица параметров: %d записей, key=%s, filename=%s",
        len(params),
        key,
        filename,
    )

    return ParameterTableUploadResponse(
        status="loaded",
        parameter_count=len(params),
        message=f"Загружено {len(params)} параметров из {filename}",
    )


@router.get("/parameter-table/{video_id}", response_model=list[ParameterTableEntry])
async def get_parameter_table(video_id: str) -> list[ParameterTableEntry]:
    """Возвращает загруженную таблицу параметров для видео.

    Args:
        video_id: ID видео.

    Returns:
        Список записей таблицы параметров.
    """
    with _parameter_table_lock:
        # Сначала ищем по video_id, затем глобальную
        params = _parameter_tables.get(video_id) or _parameter_tables.get("__global__")

    if not params:
        raise HTTPException(
            status_code=404,
            detail="Таблица параметров не загружена для данного видео",
        )

    return [
        ParameterTableEntry(
            id=p.get("id", 0),
            name=p.get("name", ""),
            unit=p.get("unit", ""),
            short_name=p.get("short_name", ""),
            decimal_places=p.get("decimal_places", 1),
            sheet_name=p.get("sheet_name", ""),
        )
        for p in params[:200]
    ]


# ---------------------------------------------------------------------------
# Дополнительные утилиты
# ---------------------------------------------------------------------------


def get_current_video_id() -> str:
    """Возвращает ID текущего обрабатываемого видео."""
    with _state_lock:
        return _pipeline_state.video_id


def is_pipeline_running() -> bool:
    """Проверяет, запущен ли конвейер."""
    with _state_lock:
        return _pipeline_state.status in ("starting", "processing")
