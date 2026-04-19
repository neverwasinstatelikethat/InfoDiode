"""API-роутер для работы с видеозаписями."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile

from app.config import settings
from app.core.video_ingestion import (
    detect_video_type,
    generate_video_id,
    get_video_info,
)
from app.models.schemas import VideoUploadResponse

router = APIRouter()


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile) -> VideoUploadResponse:
    """Загружает видеофайл и возвращает информацию о нём.

    Args:
        file: Загружаемый видеофайл.

    Returns:
        Информация о загруженном видео.
    """
    video_id = generate_video_id()
    input_dir = Path(settings.input_videos_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    dest_path = input_dir / f"{video_id}_{file.filename}"
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    info = get_video_info(dest_path)

    # Определяем тип видео по первому кадру
    import cv2

    cap = cv2.VideoCapture(str(dest_path))
    ret, frame = cap.read()
    cap.release()

    video_type = detect_video_type(frame) if ret else "direct"

    # resolution может быть tuple или string
    resolution = info["resolution"]
    if isinstance(resolution, (tuple, list)):
        resolution = f"{resolution[0]}x{resolution[1]}"

    return VideoUploadResponse(
        video_id=video_id,
        filename=file.filename or "unknown",
        video_type=video_type,
        resolution=resolution,
        fps=info["fps"],
        duration_s=info["duration_s"],
        total_frames=info["total_frames"],
    )


@router.get("/info/{video_id}")
async def get_video_details(video_id: str) -> dict:
    """Возвращает информацию о загруженном видео.

    Args:
        video_id: Идентификатор видео.

    Returns:
        Словарь с информацией о видео.
    """
    input_dir = Path(settings.input_videos_dir)
    files = list(input_dir.glob(f"{video_id}_*"))

    if not files:
        return {"error": "Видео не найдено"}

    info = get_video_info(files[0])
    return {"video_id": video_id, **info}
