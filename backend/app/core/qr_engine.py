"""Генератор QR-кодов v40 (177×177 модулей) для info-diode.

Следует best practices из отчёта qr_code.txt:
- Версия QR = 40 (фиксированный размер 177×177 модулей)
- scale=1, border=0 → точный размер 177×177 px
- Error correction = H (30% восстановления) для камеры
- Компактный JSON с separators=(',', ':')
- Тёмные модули на светлом фоне (без инверсии)

Кодирование: данные -> JSON -> QR v40-H
При наложении на видео QR масштабируется с NEAREST интерполяцией.
"""

from __future__ import annotations

import base64
import json
import logging
import zlib
from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_H

from app.models.schemas import SnapshotData

logger = logging.getLogger("infodiode")

# Цвета QR по best practices #4,#7 из отчёта qr_code.txt
# #0a2540 — стильный тёмно-синий с высоким контрастом на белом фоне
QR_DARK_COLOR = "#0a2540"
QR_LIGHT_COLOR = "#ffffff"

# QR v40 = 177x177 модулей (фиксированный по требованиям задачи)
QR_VERSION = 40
QR_MODULES = 177

# Префикс для сжатого формата INFODIODE:<base64> (для больших данных)
QR_PREFIX = "INFODIODE:"


def _prepare_json_data(params: dict[int, str], timestamp: str) -> str:
    """Подготавливает компактную JSON-строку для QR-кода.

    Следует best practice #6: JSON с separators=(',', ':') для минимального размера.
    ISO timestamp с миллисекундами.

    Args:
        params: Словарь param_id -> значение.
        timestamp: Таймстемп в формате HH:MM:SS.mmm.

    Returns:
        Компактная JSON-строка.
    """
    data = {
        **{str(k): v for k, v in params.items()},
        "ts": timestamp,
    }
    return json.dumps(data, separators=(',', ':'), ensure_ascii=False)


def _decode_payload(payload: str) -> dict:
    """Декодирует JSON-строку из QR-кода.

    Поддерживает два формата:
    1. Raw JSON (формат по отчёту qr_code.txt)
    2. INFODIODE:<base64> (сжатый формат для больших данных)

    Args:
        payload: Строка из QR-кода.

    Returns:
        Декодированный словарь.

    Raises:
        ValueError: Если данные некорректны.
    """
    # Пробуем raw JSON сначала (основной формат по отчёту)
    if payload.startswith('{'):
        return json.loads(payload)

    # Обратная совместимость: INFODIODE:<base64> формат
    if payload.startswith(QR_PREFIX):
        encoded = payload[len(QR_PREFIX):]
        decoded = base64.b64decode(encoded)
        decompressed = zlib.decompress(decoded)
        return json.loads(decompressed.decode('utf-8'))

    raise ValueError(f"Некорректный формат QR: ожидается JSON или '{QR_PREFIX}'")


def encode_snapshot_to_qr(snapshot: SnapshotData) -> qrcode.QRCode:
    """Кодирует данные снимка в QR-код v40-H.

    Следует best practices #1, #2, #3 из отчёта:
    version=40, ERROR_CORRECT_H (30% восстановления), box_size=1, border=0.

    Args:
        snapshot: Данные снимка (один 500мс интервал).

    Returns:
        Объект QR-кода (177×177 px при scale=1).
    """
    json_str = _prepare_json_data(snapshot.params, snapshot.timestamp)
    return _make_qr(json_str)


def save_qr_image(qr: qrcode.QRCode, filepath: str | Path) -> Path:
    """Сохраняет QR-код как изображение.

    Args:
        qr: Объект QR-кода.
        filepath: Путь для сохранения.

    Returns:
        Путь к сохранённому файлу.
    """
    filepath = Path(filepath)
    img = qr.make_image(fill_color=QR_DARK_COLOR, back_color=QR_LIGHT_COLOR)
    img.save(str(filepath))
    return filepath


def _make_qr(data: str) -> qrcode.QRCode:
    """Создаёт QR-код v40 с ERROR_CORRECT_H.

    Следует best practices #1, #2, #3 из отчёта qr_code.txt:
    - version=40 (фиксированный, 177×177 модулей)
    - box_size=1 (1 модуль = 1 пиксель, точный размер 177×177 px)
    - border=0 (quiet zone обеспечивается подложкой при наложении)
    - ERROR_CORRECT_H (30% восстановления для камеры)

    Args:
        data: Строка данных для кодирования.

    Returns:
        Объект QR-кода (177×177 px).

    Raises:
        ValueError: Если данные не помещаются в QR v40-H.
    """
    qr = qrcode.QRCode(
        version=QR_VERSION,
        error_correction=ERROR_CORRECT_H,
        box_size=1,  # 1 модуль = 1 пиксель → точный размер 177×177 px
        border=0,   # Quiet zone обеспечивается белой подложкой при наложении
    )
    try:
        qr.add_data(data)
        qr.make(fit=False)
    except qrcode.exceptions.DataOverflowError as e:
        raise ValueError(
            f"Данные ({len(data)} символов) не помещаются в QR v40-H "
            f"(максимум ~1852 байт). Используйте меньше параметров."
        ) from e
    logger.debug(
        "QR v%d создан: %dx%d модулей, data_len=%d",
        qr.version, qr.modules_count, qr.modules_count, len(data)
    )
    return qr


def decode_qr_to_snapshot(qr_data: str) -> dict:
    """Декодирует данные из QR-кода (для верификации).

    Args:
        qr_data: Строка данных из QR-кода (с префиксом INFODIODE:).

    Returns:
        Декодированные данные.
    """
    return _decode_payload(qr_data)


def encode_data_for_qr(data: str) -> str:
    """Кодирует строковые данные в формат для QR-кода.

    Для XML данных использует сжатый формат INFODIODE:<base64>,
    т.к. XML может превышать ёмкость QR v40-H.

    Args:
        data: Исходная строка данных (например, XML).

    Returns:
        Строка с префиксом INFODIODE:, готовая для QR-кода.
    """
    payload = {"xml": data}
    json_str = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')
    compressed = zlib.compress(json_bytes, level=9)
    encoded = base64.b64encode(compressed).decode('ascii')
    return f"{QR_PREFIX}{encoded}"


def decode_data_from_qr(encoded: str) -> str:
    """Декодирует данные из QR-кода обратно в строку.

    Поддерживает оба формата:
    1. Raw JSON — возвращает JSON-строку
    2. INFODIODE:<base64> — распаковывает и возвращает XML

    Args:
        encoded: Строка из QR-кода.

    Returns:
        Декодированная строка.

    Raises:
        ValueError: Если данные некорректны.
    """
    data = _decode_payload(encoded)
    return data.get("xml", json.dumps(data, ensure_ascii=False))


def encode_params_dict_for_qr(params: dict[int, str], timestamp: str) -> qrcode.QRCode:
    """Кодирует словарь параметров в QR-код v40-H.

    Данные кодируются как компактный JSON (best practice #6).
    Формат: {"1":"12.5","2":"3.14",...,"ts":"00:01:00.500"}

    Args:
        params: Словарь param_id -> значение.
        timestamp: Таймстемп в формате HH:MM:SS.mmm.

    Returns:
        Объект QR-кода v40-H (177×177 px при scale=1).
    """
    json_str = _prepare_json_data(params, timestamp)
    return _make_qr(json_str)
