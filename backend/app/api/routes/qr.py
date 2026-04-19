"""API роуты для QR info-diode (генерация оверлей видео).

Эндпоинты для создания видео с наложенными QR-кодами,
содержащими параметры SCADA из XML.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.core.qr_overlay import generate_overlay_video, get_overlay_status
from app.core.qr_engine import decode_qr_to_snapshot, QR_PREFIX, QR_VERSION, _decode_payload

logger = logging.getLogger("infodiode")

router = APIRouter()


def _find_source_video(video_id: str) -> Path | None:
    """Находит исходный видеофайл по video_id.

    Args:
        video_id: ID видео.

    Returns:
        Путь к видеофайлу или None если не найден.
    """
    input_dir = Path(settings.input_videos_dir)

    # Ищем файл по шаблону {video_id}_*.mp4
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        pattern = f"{video_id}_*{ext}"
        matches = list(input_dir.glob(pattern))
        if matches:
            return matches[0]

        # Пробуем точное совпадение
        exact = input_dir / f"{video_id}{ext}"
        if exact.exists():
            return exact

    return None


def _get_xml_path(video_id: str) -> Path | None:
    """Получает путь к XML файлу для video_id.

    Args:
        video_id: ID видео.

    Returns:
        Путь к XML файлу или None если не найден.
    """
    output_dir = Path(settings.output_xml_dir)
    xml_path = output_dir / f"{video_id}_output.xml"

    if xml_path.exists():
        return xml_path

    # Пробуем альтернативные имена
    alt_path = output_dir / f"{video_id}.xml"
    if alt_path.exists():
        return alt_path

    return None


def _get_overlay_path(video_id: str) -> Path:
    """Получает путь к оверлей видео для video_id.

    Args:
        video_id: ID видео.

    Returns:
        Путь к оверлей видео.
    """
    output_dir = Path(settings.output_xml_dir)
    return output_dir / f"{video_id}_qr_overlay.mp4"


@router.post("/generate/{video_id}")
async def generate_qr_overlay(video_id: str) -> dict:
    """Генерирует оверлей видео с QR-кодами для указанного video_id.

    Читает XML из data/output_xml/{video_id}_output.xml,
    исходное видео из data/input_videos/,
    сохраняет результат в data/output_xml/{video_id}_qr_overlay.mp4.

    Args:
        video_id: ID видео для обработки.

    Returns:
        Статус генерации и пути к файлам.

    Raises:
        HTTPException: Если исходные файлы не найдены или ошибка генерации.
    """
    logger.info("Запрос на генерацию QR оверлея: %s", video_id)

    # Проверяем наличие XML
    xml_path = _get_xml_path(video_id)
    if not xml_path:
        raise HTTPException(
            status_code=404,
            detail=f"XML файл не найден для video_id={video_id}. "
                   f"Ожидается: {video_id}_output.xml в {settings.output_xml_dir}"
        )

    # Находим исходное видео
    video_path = _find_source_video(video_id)
    if not video_path:
        raise HTTPException(
            status_code=404,
            detail=f"Исходное видео не найдено для video_id={video_id}"
        )

    # Определяем путь для выходного файла
    output_path = _get_overlay_path(video_id)

    logger.info(
        "Генерация QR оверлея:\n  Видео: %s\n  XML: %s\n  Выход: %s",
        video_path, xml_path, output_path
    )

    try:
        # Генерируем оверлей видео
        result_path = generate_overlay_video(
            video_path=video_path,
            xml_path=xml_path,
            output_path=output_path,
            # qr_size=177 по умолчанию (требование задания)
        )

        return {
            "status": "success",
            "video_id": video_id,
            "source_video": str(video_path),
            "source_xml": str(xml_path),
            "output_video": str(result_path),
            "message": "QR оверлей видео успешно сгенерировано",
        }

    except FileNotFoundError as e:
        logger.error("Файл не найден: %s", e)
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error("Ошибка валидации: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error("Ошибка выполнения: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Неожиданная ошибка при генерации QR оверлея")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка генерации QR оверлея: {str(e)}"
        )


@router.get("/video/{video_id}")
async def get_qr_overlay_video(video_id: str) -> FileResponse:
    """Возвращает сгенерированное оверлей видео с QR-кодами.

    Args:
        video_id: ID видео.

    Returns:
        FileResponse с видеофайлом.

    Raises:
        HTTPException: Если видео не найдено.
    """
    overlay_path = _get_overlay_path(video_id)

    if not overlay_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"QR оверлей видео не найдено для video_id={video_id}. "
                   f"Сначала вызовите POST /qr/generate/{video_id}"
        )

    return FileResponse(
        path=overlay_path,
        media_type="video/mp4",
        filename=f"{video_id}_qr_overlay.mp4",
    )


@router.get("/status/{video_id}")
async def get_qr_status(video_id: str) -> dict:
    """Проверяет статус QR оверлей видео для указанного video_id.

    Args:
        video_id: ID видео.

    Returns:
        Информация о наличии и пути к оверлей видео.
    """
    status = get_overlay_status(video_id, settings.output_xml_dir)

    # Также проверяем наличие исходных файлов
    xml_path = _get_xml_path(video_id)
    video_path = _find_source_video(video_id)

    return {
        "video_id": video_id,
        "overlay_exists": status["exists"],
        "overlay_path": status["path"],
        "overlay_size_bytes": status["size_bytes"],
        "overlay_created_at": status["created_at"],
        "source_xml_exists": xml_path is not None,
        "source_xml_path": str(xml_path) if xml_path else None,
        "source_video_exists": video_path is not None,
        "source_video_path": str(video_path) if video_path else None,
        "can_generate": xml_path is not None and video_path is not None,
        "qr_format": "JSON (compact) в QR v40-H",
        "qr_version": QR_VERSION,
        "qr_modules": 177,
        "qr_error_correction": "H (30% recovery)",
        "qr_data_encoding": "JSON с separators=(',', ':')",
    }


@router.post("/decode")
async def decode_qr_data(payload: dict) -> dict:
    """Декодирует данные из QR-кода для верификации.

    Поддерживает два формата:
    1. Raw JSON (основной формат по отчёту qr_code.txt) — сканеры читают напрямую.
    2. INFODIODE:<base64> (сжатый формат для XML данных).

    Args:
        payload: Словарь с полем qr_data — строка из QR-кода.

    Returns:
        Декодированные данные (timestamp + параметры).

    Raises:
        HTTPException: Если данные некорректны.
    """
    qr_data = payload.get("qr_data", "")
    if not qr_data:
        raise HTTPException(status_code=400, detail="Поле qr_data обязательно")

    try:
        decoded = _decode_payload(qr_data)
        # Определяем формат для ответа
        fmt = "JSON" if qr_data.startswith('{') else "INFODIODE"
        return {
            "status": "success",
            "format": fmt,
            "data": decoded,
        }
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка декодирования QR: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка декодирования QR: {str(e)}"
        )
