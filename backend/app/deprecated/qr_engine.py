"""Генератор QR-кодов v40 (177x177) для info-diode.

Кодирование: данные -> msgpack -> zlib -> base64 -> QR v40
Максимальная ёмкость: 7,089 цифр / 4,296 алфавитно-цифровых символов.
"""

from __future__ import annotations

import base64
import io
import zlib
from pathlib import Path

import msgpack
import qrcode
from qrcode.constants import ERROR_CORRECT_L

from app.models.schemas import SnapshotData


def encode_snapshot_to_qr(snapshot: SnapshotData) -> qrcode.QRCode:
    """Кодирует данные снимка в QR-код v40.

    Args:
        snapshot: Данные снимка (один 500мс интервал).

    Returns:
        Объект QR-кода.
    """
    # Подготовка данных
    data = {
        "ts": snapshot.timestamp,
        "p": {str(k): v for k, v in snapshot.params.items()},
    }

    # Сериализация msgpack
    packed = msgpack.packb(data, use_bin_type=True)

    # Сжатие zlib
    compressed = zlib.compress(packed, level=9)

    # Base64 для QR
    encoded = base64.b64encode(compressed).decode("ascii")

    # Генерация QR v40
    qr = qrcode.QRCode(
        version=40,
        error_correction=ERROR_CORRECT_L,
        box_size=4,
        border=2,
    )
    qr.add_data(encoded)
    qr.make(fit=False)

    return qr


def save_qr_image(qr: qrcode.QRCode, filepath: str | Path) -> Path:
    """Сохраняет QR-код как изображение.

    Args:
        qr: Объект QR-кода.
        filepath: Путь для сохранения.

    Returns:
        Путь к сохранённому файлу.
    """
    filepath = Path(filepath)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(str(filepath))
    return filepath


def decode_qr_to_snapshot(qr_data: str) -> dict:
    """Декодирует данные из QR-кода (для верификации).

    Args:
        qr_data: Строка данных из QR-кода.

    Returns:
        Декодированные данные.
    """
    decoded = base64.b64decode(qr_data)
    decompressed = zlib.decompress(decoded)
    return msgpack.unpackb(decompressed, raw=False)


def encode_data_for_qr(data: str) -> str:
    """Кодирует строковые данные для QR-кода.

    Кодирование: данные -> bytes -> msgpack -> zlib -> base64

    Args:
        data: Исходная строка данных (например, XML).

    Returns:
        Base64-encoded строка, готовая для QR-кода.
    """
    # Кодируем строку в bytes
    data_bytes = data.encode("utf-8")

    # Сериализация msgpack
    packed = msgpack.packb(data_bytes, use_bin_type=True)

    # Сжатие zlib
    compressed = zlib.compress(packed, level=9)

    # Base64 для QR
    encoded = base64.b64encode(compressed).decode("ascii")

    return encoded


def decode_data_from_qr(encoded: str) -> str:
    """Декодирует данные из QR-кода обратно в строку.

    Декодирование: base64 -> zlib -> msgpack -> bytes -> str

    Args:
        encoded: Base64-encoded строка из QR-кода.

    Returns:
        Исходная декодированная строка.

    Raises:
        Exception: Если данные некорректны.
    """
    # Base64 decode
    decoded = base64.b64decode(encoded)

    # Zlib decompress
    decompressed = zlib.decompress(decoded)

    # Msgpack unpack
    data_bytes = msgpack.unpackb(decompressed, raw=False)

    # Bytes to string
    return data_bytes.decode("utf-8")
