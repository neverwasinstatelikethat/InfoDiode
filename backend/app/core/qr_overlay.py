"""Генератор видео с QR-оверлеем (info-diode).

Накладывает QR-коды на каждый кадр видео,
создавая визуальный канал передачи данных.
QR размещается адаптивно — в области кадра с минимальным
количеством текста/контента (незанятый участок мнемосхемы).
Размер QR: 177×177 px по требованию задания.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import qrcode

from app.core.qr_engine import encode_params_dict_for_qr, QR_DARK_COLOR, QR_LIGHT_COLOR

logger = logging.getLogger("infodiode")

# Отступ от краёв кадра до QR-кода (в пикселях)
QR_MARGIN = 4

# Размер QR по требованию задания (QR v40 = 177×177 модулей = 177×177 px)
QR_DEFAULT_SIZE = 177



def _find_free_region(frame: np.ndarray, qr_size: int, grid_rows: int = 8, grid_cols: int = 8) -> tuple[int, int]:
    """Находит область кадра с минимальным содержимым для размещения QR.

    Анализирует текстовую плотность (по краям и дисперсии) в каждой ячейке сетки
    и выбирает позицию с наименьшим количеством контента.
    Это соответствует требованию задания: «Место выбрать адаптивно,
    исходя из наличия незанятого участка мнемосхемы».

    Args:
        frame: Кадр видео (BGR).
        qr_size: Размер QR-кода в пикселях.
        grid_rows: Количество строк сетки.
        grid_cols: Количество столбцов сетки.

    Returns:
        Координаты (x, y) левого верхнего угла для размещения QR.
    """
    height, width = frame.shape[:2]

    # Конвертируем в grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детектор рёбер — высокая плотность рёбер = много текста/контента
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.resize(edges, (grid_cols, grid_rows), interpolation=cv2.INTER_AREA)
    edge_density = edge_density.astype(np.float32) / 255.0

    cell_h = height // grid_rows
    cell_w = width // grid_cols

    # Сколько ячеек занимает QR
    qr_cells_h = max(1, round(qr_size / cell_h))
    qr_cells_w = max(1, round(qr_size / cell_w))

    best_score = float('inf')
    best_x, best_y = QR_MARGIN, QR_MARGIN  # fallback

    for i in range(grid_rows - qr_cells_h + 1):
        for j in range(grid_cols - qr_cells_w + 1):
            # Суммарная плотность рёбер в области размещения
            region = edge_density[i:i + qr_cells_h, j:j + qr_cells_w]
            score = float(np.sum(region))

            if score < best_score:
                best_score = score
                best_x = max(QR_MARGIN, j * cell_w)
                best_y = max(QR_MARGIN, i * cell_h)

    # Убеждаемся, что QR не выходит за пределы кадра
    if best_x + qr_size > width:
        best_x = max(QR_MARGIN, width - qr_size - QR_MARGIN)
    if best_y + qr_size > height:
        best_y = max(QR_MARGIN, height - qr_size - QR_MARGIN)

    logger.info(
        "Адаптивная позиция QR: (%d, %d), score=%.3f (минимальная плотность рёбер)",
        best_x, best_y, best_score
    )
    return (best_x, best_y)


def _parse_xml_snapshots(xml_path: str | Path) -> list[dict]:
    """Парсит XML файл и извлекает снапшоты параметров.

    Args:
        xml_path: Путь к XML файлу.

    Returns:
        Список словарей с timestamp и params.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML файл не найден: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    snapshots = []
    for params_elem in root.findall("parameters"):
        timestamp = params_elem.get("timestamp", "")
        # Убираем пробелы вокруг значения timestamp
        timestamp = timestamp.strip() if timestamp else ""

        params = {}
        for param_elem in params_elem.findall("param"):
            param_id = param_elem.get("id")
            if param_id:
                params[int(param_id)] = param_elem.text or ""

        snapshots.append({
            "timestamp": timestamp,
            "params": params,
        })

    return snapshots


def generate_overlay_video(
    video_path: str | Path,
    xml_path: str | Path,
    output_path: str | Path,
    qr_size: int = QR_DEFAULT_SIZE,
) -> Path:
    """Генерирует видео с QR-оверлеем поверх оригинальных кадров.

    QR-код 177×177 px (по требованию задания) размещается адаптивно —
    в области с минимальным количеством текста/контента.
    Каждый 500мс QR обновляется с новыми параметрами из XML.

    Args:
        video_path: Путь к исходному видеофайлу.
        xml_path: Путь к XML файлу с параметрами.
        output_path: Путь к выходному видеофайлу.
        qr_size: Размер QR-кода в пикселях. По умолчанию 177 (требование задания).

    Returns:
        Путь к сгенерированному видео.
    """
    video_path = Path(video_path)
    xml_path = Path(xml_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Видео файл не найден: {video_path}")

    # Парсим XML
    logger.info("Парсинг XML: %s", xml_path)
    snapshots = _parse_xml_snapshots(xml_path)
    logger.info("Найдено снапшотов: %d", len(snapshots))

    if not snapshots:
        raise ValueError("В XML файле не найдено снапшотов параметров")

    # Открываем видео
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "Видео: %dx%d, %.2f fps, %d кадров",
        width, height, fps, total_frames
    )

    # Размер QR: 177×177 px по требованию задания
    logger.info("Размер QR-кода: %d×%d px", qr_size, qr_size)

    # Создаём VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать выходной файл: {output_path}")

    # Читаем первый кадр для адаптивного позиционирования
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise RuntimeError("Не удалось прочитать первый кадр")

    # Адаптивное размещение: ищем область с минимальной плотностью контента
    qr_x, qr_y = _find_free_region(first_frame, qr_size)

    # Возвращаемся к началу видео
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Параметры для синхронизации
    interval_frames = int(fps * 0.5)  # 500мс
    snapshot_idx = 0
    current_qr_array = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Определяем, нужен ли новый QR на этом кадре
        if frame_idx % interval_frames == 0 and snapshot_idx < len(snapshots):
            snapshot = snapshots[snapshot_idx]
            timestamp = snapshot["timestamp"]
            params = snapshot["params"]

            if params:
                # Генерируем QR-код v40-H (177×177 px при scale=1)
                qr = encode_params_dict_for_qr(params, timestamp)
                qr_img = qr.make_image(fill_color=QR_DARK_COLOR, back_color=QR_LIGHT_COLOR)
                qr_array = np.array(qr_img.convert("RGB"))
                qr_array = cv2.cvtColor(qr_array, cv2.COLOR_RGB2BGR)

                # 177×177 — масштабирование не требуется (совпадает с qr_size)
                if qr_array.shape[0] != qr_size or qr_array.shape[1] != qr_size:
                    current_qr_array = cv2.resize(
                        qr_array, (qr_size, qr_size), interpolation=cv2.INTER_NEAREST
                    )
                else:
                    current_qr_array = qr_array

            snapshot_idx += 1

        # Накладываем QR на кадр напрямую (без подложки)
        if current_qr_array is not None:
            end_y = min(qr_y + qr_size, height)
            end_x = min(qr_x + qr_size, width)
            if end_y > qr_y and end_x > qr_x:
                frame[qr_y:end_y, qr_x:end_x] = current_qr_array[:end_y - qr_y, :end_x - qr_x]

        out.write(frame)
        frame_idx += 1

        # Логируем прогресс каждые 100 кадров
        if frame_idx % 100 == 0:
            progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            logger.debug("Прогресс: %.1f%% (%d/%d кадров)", progress, frame_idx, total_frames)

    cap.release()
    out.release()

    logger.info(
        "Видео с QR-оверлеем сохранено: %s (%d кадров обработано)",
        output_path, frame_idx
    )

    return output_path


def get_overlay_status(video_id: str, output_xml_dir: str | Path) -> dict:
    """Проверяет статус QR оверлей видео.

    Args:
        video_id: ID видео.
        output_xml_dir: Директория с выходными файлами.

    Returns:
        Словарь со статусом: exists, path, created_at.
    """
    output_xml_dir = Path(output_xml_dir)
    overlay_path = output_xml_dir / f"{video_id}_qr_overlay.mp4"

    import os
    stat = overlay_path.stat() if overlay_path.exists() else None

    return {
        "exists": overlay_path.exists(),
        "path": str(overlay_path) if overlay_path.exists() else None,
        "created_at": stat.st_mtime if stat else None,
        "size_bytes": stat.st_size if stat else None,
    }
