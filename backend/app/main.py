"""Точка входа FastAPI-приложения InfoDiode.

Поддерживает два режима запуска:
1. Локальный:  python -m uvicorn app.main:app --reload
2. Docker:     CMD ["python3.12", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

Инициализация компонентов (lifespan):
- Создание директорий данных
- Настройка GPG-ключа
- Проверка доступности VLM-сервера (llama-server)
- Проверка доступности Redis (не блокирует запуск)
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, evaluation, pipeline, qr, video
from app.api.websocket import ws_manager
from app.config import settings

logger = logging.getLogger("infodiode")


# ---------------------------------------------------------------------------
# Инициализация компонентов
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    """Настраивает логирование.

    Включает логи для всех модулей app.* с правильным форматированием.
    """
    level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # Перезаписывает существующие обработчики
    )

    # Убеждаемся, что все логи app.* имеют правильный уровень
    logging.getLogger("app").setLevel(level)
    logging.getLogger("app.core").setLevel(level)
    logging.getLogger("app.api").setLevel(level)

    # Отключаем избыточные логи от библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info("Логирование: уровень=%s", logging.getLevelName(level))


def _ensure_directories() -> None:
    """Создаёт все необходимые директории данных."""
    dirs = [
        settings.input_videos_dir,
        settings.output_xml_dir,
        settings.qr_codes_dir,
        settings.encryption_keys_dir,
        settings.models_dir,
        Path(settings.output_xml_dir).parent / "calibration",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Директории данных: %s",
        Path(settings.input_videos_dir).parent,
    )


def _setup_gpg_key() -> None:
    """Создаёт GPG-ключ для шифрования, если он не существует."""
    import gnupg

    gpg_home = Path(settings.gpg_home)
    gpg_home.mkdir(parents=True, exist_ok=True)

    gpg = gnupg.GPG(gnupghome=str(gpg_home))

    # Проверяем наличие ключа
    existing_keys = gpg.list_keys()
    for key in existing_keys:
        for uid in key.get("uids", []):
            if settings.gpg_recipient in uid:
                logger.info("GPG-ключ найден: %s", uid)
                return

    # Генерируем новый ключ (batch mode, без пароля)
    logger.info("Генерация GPG-ключа для %s...", settings.gpg_recipient)
    input_data = gpg.gen_key_input(
        key_type="RSA",
        key_length=4096,
        subkey_type="RSA",
        subkey_length=4096,
        name_real="InfoDiode",
        name_email=settings.gpg_recipient,
        expire_date=0,
        no_protection=True,
    )
    key = gpg.gen_key(input_data)
    if key:
        logger.info("GPG-ключ создан: %s", key.fingerprint)
    else:
        logger.warning("Не удалось создать GPG-ключ — шифрование будет недоступно")


async def _check_vlm_server() -> bool:
    """Проверяет доступность VLM-сервера (llama-server).

    Returns:
        True если сервер доступен, иначе False.
    """
    try:
        from app.core.vlm_client import get_vlm_client

        client = get_vlm_client()
        available = await client.health_check()
        await client.close()

        if available:
            logger.info("VLM сервер доступен: %s", settings.vlm_base_url)
            return True
        else:
            logger.warning(
                "VLM сервер не отвечает: %s",
                settings.vlm_base_url,
            )
            logger.info(
                "  → Запустите llama-server: .\\llama-server.exe -m <model.gguf> "
                "--port 8090 --host 0.0.0.0 --mmproj <mmproj.gguf>"
            )
            return False

    except ImportError:
        logger.warning("Модуль VLM client не установлен")
        return False
    except Exception as e:
        logger.warning("Не удалось проверить VLM сервер: %s", e)
        return False


def _check_redis() -> bool:
    """Проверяет доступность Redis (не блокирует запуск)."""
    try:
        import redis

        r = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        r.close()
        logger.info("Redis доступен: %s", settings.redis_url)
        return True
    except ImportError:
        logger.warning("Модуль redis не установлен — Celery будет недоступен")
        return False
    except Exception as e:
        logger.warning("Redis недоступен (%s): %s", settings.redis_url, e)
        logger.info("  → Запустите Redis: docker run -p 6379:6379 redis:7-alpine")
        return False


def _check_smtp() -> bool:
    """Проверяет доступность SMTP-сервера (не блокирует запуск)."""
    import smtplib

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=3) as server:
            server.noop()
        logger.info("SMTP доступен: %s:%s", settings.smtp_host, settings.smtp_port)
        return True
    except Exception as e:
        logger.warning("SMTP недоступен (%s:%s): %s", settings.smtp_host, settings.smtp_port, e)
        if settings.is_local:
            logger.info(
                "  → Запустите Mailpit: docker run -p 1025:1025 -p 8025:8025 axllent/mailpit"
            )
        return False


def _print_startup_banner() -> None:
    """Выводит баннер с информацией о запуске."""
    mode = "ЛОКАЛЬНЫЙ" if settings.is_local else "DOCKER"
    logger.info("=" * 60)
    logger.info("  InfoDiode v0.2.0 — VLM Pipeline — Режим: %s", mode)
    logger.info("=" * 60)
    logger.info("  API:       http://localhost:8000")
    logger.info("  Docs:      http://localhost:8000/docs")
    logger.info("  WebSocket: ws://localhost:8000/ws")
    logger.info("  Data:      %s", Path(settings.input_videos_dir).parent)
    logger.info("  VLM URL:   %s", settings.vlm_base_url)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Lifespan (инициализация / завершение)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения: startup / shutdown."""
    # === STARTUP ===
    _setup_logging()
    _print_startup_banner()
    _ensure_directories()

    # GPG-ключ (не блокирует, но логирует предупреждение)
    try:
        _setup_gpg_key()
    except ImportError:
        logger.warning("python-gnupg не установлен — GPG-шифрование недоступно")
        logger.info("  → Установите GPG: https://gnupg.org/download/ и pip install python-gnupg")
    except Exception as e:
        logger.warning("Ошибка настройки GPG: %s", e)

    # Проверка VLM-сервера (не блокирует)
    import time
    startup_start = time.perf_counter()

    logger.info("Проверка VLM-сервера...")
    vlm_ok = await _check_vlm_server()

    # Проверка Redis (не блокирует)
    redis_ok = _check_redis()

    # Проверка SMTP (не блокирует)
    _check_smtp()

    # Информационное сообщение
    if settings.is_local:
        logger.info("Локальный запуск: pipeline работает в фоне через asyncio.create_task")
        if not redis_ok:
            logger.info("  Celery worker недоступен — задачи выполняются синхронно")
        if not vlm_ok:
            logger.warning("  VLM сервер недоступен — pipeline не будет работать корректно")

    # Создание администратора по умолчанию
    from app.core.auth_service import create_default_admin
    create_default_admin()

    total_startup = time.perf_counter() - startup_start
    logger.info(
        "InfoDiode готов к работе (всего: %.2fs, VLM: %s)",
        total_startup,
        "доступен" if vlm_ok else "недоступен",
    )

    yield  # Приложение работает

    # === SHUTDOWN ===
    logger.info("InfoDiode завершает работу...")


# ---------------------------------------------------------------------------
# FastAPI-приложение
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    description="Air-gapped data extraction system: SCADA video VLM analysis → XML → GPG → SMTP",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS для фронтенда (офлайн-first, JWT через Authorization header — cookies не нужны)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Роутеры
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(video.router, prefix="/api/video", tags=["video"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(qr.router, prefix="/api/qr", tags=["qr"])


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Проверка работоспособности сервиса."""
    return {"status": "ok", "app": settings.app_name}


@app.get("/")
async def root() -> dict[str, str]:
    """Корневой эндпоинт."""
    return {
        "app": settings.app_name,
        "version": "0.2.0",
        "mode": "local" if settings.is_local else "docker",
        "pipeline": "vlm",
        "docs": "/docs",
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket-эндпоинт для real-time обновлений."""
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Прямой запуск: python -m app.main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    _setup_logging()
    logger.info("Запуск InfoDiode через python -m app.main ...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
