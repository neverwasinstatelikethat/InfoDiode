#!/usr/bin/env python3
"""Локальный запуск InfoDiode без Docker.

Использование:
    python run_local.py                  # С предзагрузкой OCR (медленный старт)
    INFODIODE_PRELOAD_OCR=false python run_local.py  # Быстрый старт, OCR загрузится при первом запросе

Зависимости (установить предварительно):
    pip install -e ".[dev]"

Дополнительно (опционально):
    - Redis:  docker run -p 6379:6379 redis:7-alpine
    - Mailpit: docker run -p 1025:1025 -p 8025:8025 axllent/mailpit
    - GPG:    https://gnupg.org/download/ + pip install python-gnupg
    - OCR:    pip install paddleocr paddlepaddle
"""

import os
import sys
from pathlib import Path

# Убеждаемся, что backend/ — в sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Устанавливаем PYTHONPATH
os.environ.setdefault("PYTHONPATH", str(backend_dir))

# Отключаем OneDNN для PaddlePaddle 3.x на Windows
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("MKLDNN_DISABLE", "1")

# PaddleOCR 2.9+ тянет modelscope -> torch. DLL torch ломается если paddle загрузится первым.
try:
    import torch  # noqa: F401
except ImportError:
    pass

# Патчим paddle.inference.Config.enable_mkldnn → no-op,
# чтобы PaddleX не включал OneDNN в inference-предикторах.
try:
    import paddle
    paddle.set_flags({"FLAGS_use_mkldnn": False})
    _cfg_cls = paddle.inference.Config
    _cfg_cls.enable_mkldnn = lambda self, *a, **kw: None
    if hasattr(_cfg_cls, "enable_mkldnn_bfloat16"):
        _cfg_cls.enable_mkldnn_bfloat16 = lambda self, *a, **kw: None
except Exception:
    pass

if __name__ == "__main__":
    import uvicorn

    # Определяем порт
    port = int(os.getenv("INFODIODE_PORT", "8000"))

    # Определяем режим reload
    reload_mode = os.getenv("INFODIODE_RELOAD", "true").lower() in ("1", "true", "yes")

    print("=" * 60)
    print("  InfoDiode — Локальный запуск")
    print("=" * 60)
    print(f"  Порт:     {port}")
    print(f"  Reload:   {reload_mode}")
    print(f"  Python:   {sys.executable}")
    print(f"  Backend:  {backend_dir}")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload_mode,
        log_level="debug" if os.getenv("INFODIODE_DEBUG", "true").lower() in ("1", "true") else "info",
    )
