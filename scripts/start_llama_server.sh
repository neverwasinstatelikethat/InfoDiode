#!/bin/bash
# Скрипт запуска llama.cpp сервера с моделью Qwen3.5-4B
#
# Использование:
#   ./scripts/start_llama_server.sh
#
# Требования:
#   - Модели должны быть загружены скриптом download_model.py
#   - llama.cpp бинарники доступны в PATH или в директории llama.cpp/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="$ROOT/backend/models/qwen3.5/Qwen3.5-4B-Q4_K_M.gguf"
MMPROJ_PATH="$ROOT/backend/models/qwen3.5/mmproj-F32.gguf"

# Определяем путь к llama-server (проверяем несколько вариантов)
if command -v llama-server &> /dev/null; then
    LLAMA_SERVER="llama-server"
elif [ -f "$ROOT/llama.cpp/llama-server" ]; then
    LLAMA_SERVER="$ROOT/llama.cpp/llama-server"
elif [ -f "$ROOT/llama.cpp/llama-server.exe" ]; then
    # WSL: используем Windows бинарник
    LLAMA_SERVER="$ROOT/llama.cpp/llama-server.exe"
else
    echo "Ошибка: llama-server не найден"
    echo "Установите llama.cpp или добавьте его в PATH"
    exit 1
fi

# Проверяем наличие моделей
if [ ! -f "$MODEL_PATH" ]; then
    echo "Ошибка: Модель не найдена: $MODEL_PATH"
    echo "Запустите: python scripts/download_model.py"
    exit 1
fi

if [ ! -f "$MMPROJ_PATH" ]; then
    echo "Ошибка: mmproj не найден: $MMPROJ_PATH"
    echo "Запустите: python scripts/download_model.py"
    exit 1
fi

echo "Запуск llama.cpp сервера..."
echo "Модель: $MODEL_PATH"
echo "mmproj: $MMPROJ_PATH"
echo "Порт: 8090"
echo

exec "$LLAMA_SERVER" \
  --model "$MODEL_PATH" \
  --mmproj "$MMPROJ_PATH" \
  --port 8090 \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --image-min-tokens 1024 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20
