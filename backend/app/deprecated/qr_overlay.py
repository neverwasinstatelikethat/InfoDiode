"""Генератор видео с QR-оверлеем (info-diode).

Накладывает QR-коды на каждый кадр видео,
создавая визуальный канал передачи данных.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import qrcode

from app.models.schemas import SnapshotData
from app.core.qr_engine import encode_snapshot_to_qr


def generate_overlay_video(
    video_path: str | Path,
    snapshots: list[SnapshotData],
    output_path: str | Path,
    qr_size: int = 200,
    qr_position: tuple[int, int] = (20, 20),
) -> Path:
    """Генерирует видео с QR-оверлеем поверх оригинальных кадров.

    Args:
        video_path: Путь к исходному видеофайлу.
        snapshots: Список снимков данных.
        output_path: Путь к выходному видеофайлу.
        qr_size: Размер QR-кода в пикселях.
        qr_position: Позиция QR (x, y) на кадре.

    Returns:
        Путь к сгенерированному видео.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    snapshot_idx = 0
    interval_frames = int(fps * 0.5)  # 500мс

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Определяем, нужен ли QR на этом кадре
        if frame_idx % interval_frames == 0 and snapshot_idx < len(snapshots):
            snapshot = snapshots[snapshot_idx]
            qr = encode_snapshot_to_qr(snapshot)
            qr_img = qr.make_image(fill_color="black", back_color="white")
            qr_array = np.array(qr_img.convert("RGB"))
            qr_array = cv2.cvtColor(qr_array, cv2.COLOR_RGB2BGR)

            # Масштабируем QR
            qr_resized = cv2.resize(qr_array, (qr_size, qr_size))

            # Накладываем QR на кадр
            x, y = qr_position
            if y + qr_size <= height and x + qr_size <= width:
                # Полупрозрачный фон
                overlay = frame.copy()
                overlay[y : y + qr_size, x : x + qr_size] = qr_resized
                cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame[y : y + qr_size, x : x + qr_size])

            snapshot_idx += 1

        out.write(frame)

    cap.release()
    out.release()

    return Path(output_path)
