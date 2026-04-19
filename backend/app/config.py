"""Конфигурация приложения InfoDiode.

Поддерживает два режима запуска:
1. Docker — переменные окружения задаются в docker-compose.yml
2. Локальный — значения по умолчанию ориентированы на рабочую копию проекта
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings

# Определяем корень проекта (на один уровень выше backend/app/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Признак локального запуска (вне Docker)
IS_LOCAL = not Path("/app/.dockerenv").exists() and os.getenv("INFODIODE_DOCKER", "").lower() not in ("1", "true", "yes")


def _local_data_dir() -> str:
    """Возвращает путь к data/ для локального запуска."""
    return str(_PROJECT_ROOT / "data")


def _local_models_dir() -> str:
    """Возвращает путь к моделям для локального запуска."""
    return str(_PROJECT_ROOT / "backend" / "models")


class Settings(BaseSettings):
    """Настройки приложения, загружаемые из переменных окружения.

    Переменные окружения с префиксом INFODIODE_ перекрывают значения по умолчанию.
    При локальном запуске (без Docker) по умолчанию используются пути
    относительно корня проекта.
    """

    # Приложение
    app_name: str = "InfoDiode"
    debug: bool = True

    # Режим запуска (автоопределение, можно задать вручную)
    is_local: bool = IS_LOCAL

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # Директории (локальные по умолчанию, Docker переопределяет через env)
    input_videos_dir: str = _local_data_dir() + "/input_videos" if IS_LOCAL else "/app/data/input_videos"
    output_xml_dir: str = _local_data_dir() + "/output_xml" if IS_LOCAL else "/app/data/output_xml"
    qr_codes_dir: str = _local_data_dir() + "/qr_codes" if IS_LOCAL else "/app/data/qr_codes"
    encryption_keys_dir: str = _local_data_dir() + "/encryption_keys" if IS_LOCAL else "/app/data/encryption_keys"
    models_dir: str = _local_models_dir() if IS_LOCAL else "/app/models"

    # VLM (Vision-Language Model) — Qwen3.5-4B через llama.cpp

    vlm_base_url: str = "http://localhost:8090"  # Базовый URL llama-server (без /v1)
    vlm_model_name: str = "Qwen3.5-4B"
    vlm_max_tokens: int = 8192  # Достаточно для больших таблиц параметров
    vlm_temperature: float = 0.1  # OCR нужна жесткая детерминированность (было 0.7)
    vlm_top_p: float = 0.9  # Qwen3.5 рекомендация: баланс разнообразия/фокуса
    vlm_top_k: int = 50  # Qwen3.5 рекомендация: ограничивает vocabulary
    vlm_presence_penalty: float = 1.5  # Qwen3.5 рекомендация: уменьшает повторения
    vlm_repetition_penalty: float = 1.0  # Qwen3.5 рекомендация: без дополнительного штрафа
    vlm_frame_interval_ms: int = 500  # Интервал обработки кадров
    vlm_concurrency: int = 1  # Один запрос за раз (KV cache limit)
    vlm_skip_similar_frames: bool = False  # Пропускать похожие кадры
    vlm_similarity_threshold: float = 0.99  # Порог схожести для пропуска
    vlm_max_image_size: int = 1920  # Optimal balance: SSIM 0.96 + text clarity + 30% less VRAM
    vlm_use_direct: bool = False  # Использовать VLMClientDirect (native) вместо HTTP

    # Zone-based analysis (сегментация кадра на зоны)
    vlm_zone_enabled: bool = True  # Включить зонный анализ (рекомендуется)
    vlm_zone_crop_padding_px: int = 15  # Padding вокруг зоны в пикселях
    vlm_zone_min_crop_size: int = 512  # Минимальный размер кропа (upscale если меньше)

    # Pipeline параллелизм
    pipeline_queue_size: int = 8  # Размер очереди между стадиями (backpressure)
    postprocess_workers: int = 2  # Количество потоков постобработки результатов OCR

    # Временные параметры
    snapshot_interval_ms: int = 500

    # SMTP (локально — localhost, в Docker — mailpit)
    smtp_host: str = "localhost" if IS_LOCAL else "mailpit"
    smtp_port: int = 1025
    smtp_from: str = "infodiode@local"

    # GPG
    gpg_home: str = _local_data_dir() + "/encryption_keys" if IS_LOCAL else "/app/data/encryption_keys"
    gpg_recipient: str = "infodiode@local"

    # JWT Аутентификация
    jwt_secret: str = "infodiode-super-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 часа

    model_config = {"env_prefix": "INFODIODE_", "env_file": ".env"}


settings = Settings()
