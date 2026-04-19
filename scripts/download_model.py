#!/usr/bin/env python3
"""Скрипт загрузки моделей Qwen3.5-4B для llama.cpp.

Загружает файлы GGUF из HuggingFace репозитория unsloth/Qwen3.5-4B-GGUF:
- Qwen3.5-4B-Q4_K_M.gguf — основная модель (~2.74 GB)
- mmproj-F32.gguf — проекция для мультимодальных возможностей (~1.33 GB)

Использование:
    python scripts/download_model.py

Требования:
    pip install huggingface_hub
"""

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


# Конфигурация
HF_REPO_ID = "unsloth/Qwen3.5-4B-GGUF"
MODEL_FILES = [
    "Qwen3.5-4B-Q4_K_M.gguf",
    "mmproj-F32.gguf",
]
# Целевая директория относительно корня проекта
TARGET_DIR = Path(__file__).resolve().parent.parent / "backend" / "models" / "qwen3.5"


def download_models() -> None:
    """Загружает все файлы моделей из HuggingFace.
    
    Создаёт целевую директорию при необходимости.
    Выводит прогресс загрузки для каждого файла.
    При ошибке загрузки выводит сообщение и завершает работу с кодом 1.
    """
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Целевая директория: {TARGET_DIR}")
    print(f"Репозиторий HuggingFace: {HF_REPO_ID}")
    print("-" * 60)
    
    for filename in MODEL_FILES:
        target_path = TARGET_DIR / filename
        
        # Пропускаем, если файл уже существует
        if target_path.exists():
            file_size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"[ПРОПУСК] {filename} уже существует ({file_size_mb:.2f} MB)")
            continue
        
        print(f"\n[ЗАГРУЗКА] {filename}...")
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=str(TARGET_DIR),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            downloaded_path = Path(downloaded_path)
            file_size_mb = downloaded_path.stat().st_size / (1024 * 1024)
            print(f"[ГОТОВО] {filename} ({file_size_mb:.2f} MB)")
        except Exception as e:
            print(f"[ОШИБКА] Не удалось загрузить {filename}: {e}", file=sys.stderr)
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Все модели успешно загружены!")
    print(f"Директория: {TARGET_DIR}")
    
    # Выводим список загруженных файлов
    print("\nЗагруженные файлы:")
    for f in TARGET_DIR.iterdir():
        if f.is_file() and f.suffix == ".gguf":
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    download_models()
