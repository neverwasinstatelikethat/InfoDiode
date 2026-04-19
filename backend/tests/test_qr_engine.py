"""Тесты для генератора QR-кодов v40-H (по отчёту qr_code.txt)."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from app.core.qr_engine import encode_data_for_qr, decode_data_from_qr, QR_PREFIX, QR_VERSION, QR_DARK_COLOR, QR_LIGHT_COLOR


def test_encode_decode_roundtrip() -> None:
    original = '<?xml version="1.0"?><sheme timestamp = "00:01:00.500"><param id="1">12.5</param></sheme>'
    encoded = encode_data_for_qr(original)
    assert isinstance(encoded, str)
    decoded = decode_data_from_qr(encoded)
    assert decoded == original


def test_encode_produces_base64() -> None:
    data = "test data 123"
    encoded = encode_data_for_qr(data)
    # encode_data_for_qr использует INFODIODE: формат для XML данных
    assert encoded.startswith(QR_PREFIX)
    import re
    base64_part = encoded[len(QR_PREFIX):]
    assert re.match(r'^[A-Za-z0-9+/=]+$', base64_part)


def test_encode_empty_string() -> None:
    encoded = encode_data_for_qr("")
    decoded = decode_data_from_qr(encoded)
    assert decoded == ""


def test_encode_unicode() -> None:
    data = "Давление P1 = 12.5 МПа"
    encoded = encode_data_for_qr(data)
    decoded = decode_data_from_qr(encoded)
    assert decoded == data


def test_decode_invalid_base64() -> None:
    # Без префикса INFODIODE:
    with pytest.raises(Exception):
        decode_data_from_qr("!!!invalid!!!")
    # С префиксом, но невалидный base64
    with pytest.raises(Exception):
        decode_data_from_qr(f"{QR_PREFIX}!!!invalid!!!")


def test_encode_large_payload() -> None:
    # Generate XML payload typical for QR v40
    params = "".join(f'<param id="{i}">{i * 1.5:.1f}</param>' for i in range(1, 50))
    data = f'<?xml version="1.0"?><sheme timestamp = "00:01:00.500">{params}</sheme>'
    encoded = encode_data_for_qr(data)
    decoded = decode_data_from_qr(encoded)
    assert decoded == data


def test_params_qr_is_v40_h() -> None:
    """QR для параметров должен быть версии 40 с ERROR_CORRECT_H."""
    from app.core.qr_engine import encode_params_dict_for_qr, _prepare_json_data
    from qrcode.constants import ERROR_CORRECT_H

    params = {1: "12.5", 2: "3.14", 3: "0.95", 4: "25.0"}
    qr = encode_params_dict_for_qr(params, "00:01:00.500")

    assert qr.version == QR_VERSION, f"Ожидалась версия {QR_VERSION}, получена {qr.version}"
    assert qr.modules_count == 177, f"Ожидалось 177 модулей, получено {qr.modules_count}"
    assert qr.error_correction == ERROR_CORRECT_H, "Ожидалась ERROR_CORRECT_H"


def test_params_qr_raw_json_readable() -> None:
    """QR содержит raw JSON — стандартные сканеры читают напрямую."""
    from app.core.qr_engine import encode_params_dict_for_qr, _decode_payload

    params = {1: "12.5", 2: "3.14"}
    qr = encode_params_dict_for_qr(params, "00:01:00.500")

    # Извлекаем данные из QR (qr.data_list возвращает bytes)
    data_bytes = b""
    for d in qr.data_list:
        data_bytes = d.data if hasattr(d, "data") else str(d).encode()

    # Конвертируем bytes в str для проверки
    data_str = data_bytes.decode('utf-8') if isinstance(data_bytes, bytes) else data_bytes

    # Должен быть raw JSON (начинается с '{')
    assert data_str.startswith('{'), f"Ожидался JSON, получено: {data_str[:30]}"

    # Декодируем и проверяем
    decoded = _decode_payload(data_str)
    assert decoded["ts"] == "00:01:00.500"
    assert decoded["1"] == "12.5"
    assert decoded["2"] == "3.14"


def test_params_qr_image_177x177() -> None:
    """QR-изображение должно быть 177×177 px (scale=1, border=0) с цветами #0a2540/#ffffff."""
    from app.core.qr_engine import encode_params_dict_for_qr

    params = {1: "12.5", 2: "3.14"}
    qr = encode_params_dict_for_qr(params, "00:01:00.500")
    img = qr.make_image(fill_color=QR_DARK_COLOR, back_color=QR_LIGHT_COLOR)

    assert img.size[0] == 177, f"Ожидалась ширина 177, получена {img.size[0]}"
    assert img.size[1] == 177, f"Ожидалась высота 177, получена {img.size[1]}"

    # Проверяем, что цвет #0a2540 используется (не чисто чёрный)
    img_rgb = img.convert("RGB")
    pixels = list(img_rgb.getdata())
    dark_pixel = (10, 37, 64)  # #0a2540
    light_pixel = (255, 255, 255)  # #ffffff
    has_dark = any(p == dark_pixel for p in pixels)
    assert has_dark, "QR должен использовать цвет #0a2540 (best practice #4,#7 из отчёта)"


def test_qr_overlay_video_generation() -> None:
    """Интеграционный тест: генерация видео с QR-оверлеем.

    Использует vid2_first_10sec.mp4 + соответствующий XML.
    Тестирует полный цикл: XML -> снапшоты -> QR-коды -> оверлей видео.
    Проверяет: файл создаётся, размер > 0, разрешение совпадает.
    Видео сохраняется в data/output_xml/qr_overlay_test.mp4.
    """
    from app.core.qr_overlay import generate_overlay_video

    # Пути к тестовым файлам
    project_root = Path(__file__).resolve().parent.parent.parent
    video_id = "5f02cd2a-e302-4da6-8604-d788fd46e234"
    video_path = project_root / "data" / "input_videos" / f"{video_id}_vid2_first_10sec.mp4"
    xml_path = project_root / "data" / "output_xml" / f"{video_id}_output.xml"
    output_path = project_root / "data" / "output_xml" / "qr_overlay_test.mp4"

    # Пропускаем если файлы не найдены
    if not video_path.exists():
        pytest.skip(f"Тестовое видео не найдено: {video_path}")
    if not xml_path.exists():
        pytest.skip(f"XML не найден: {xml_path}")

    result = generate_overlay_video(
        video_path=video_path,
        xml_path=xml_path,
        output_path=output_path,
    )

    # Проверяем, что файл создан
    assert result.exists(), f"Выходной файл не создан: {result}"

    # Проверяем размер файла > 0
    file_size = result.stat().st_size
    assert file_size > 0, "Выходной файл пуст"

    # Проверяем разрешение видео через OpenCV
    cap = cv2.VideoCapture(str(result))
    assert cap.isOpened(), "Не удалось открыть сгенерированное видео"

    out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Разрешение должно совпадать с исходным
    src_cap = cv2.VideoCapture(str(video_path))
    src_width = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_cap.release()

    assert out_width == src_width, f"Ширина не совпадает: {out_width} != {src_width}"
    assert out_height == src_height, f"Высота не совпадает: {out_height} != {src_height}"
    assert out_frames > 0, "Видео не содержит кадров"
    assert out_fps > 0, "FPS = 0"

    # Проверяем, что в первом кадре есть QR-оверлей (белая подложка)
    cap = cv2.VideoCapture(str(result))
    ret, first_frame = cap.read()
    cap.release()
    assert ret, "Не удалось прочитать первый кадр"

    # Конвертируем в grayscale и ищем белый прямоугольник (QR подложка)
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    white_mask = (gray > 250).astype(np.uint8) * 255
    white_pixels = cv2.countNonZero(white_mask)
    assert white_pixels > 0, "Не найдено белых пикселей (QR подложка)"
