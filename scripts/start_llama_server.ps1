# Скрипт запуска llama.cpp сервера с моделью Qwen3.5-4B
#
# Оптимизированный запуск для хакатона (150-450 мс/кадр на RTX 5070)
#
# Использование:
#   .\scripts\start_llama_server.ps1
#
# Требования:
#   - Модели должны быть загружены скриптом download_model.py
#   - llama.cpp бинарники в директории llama.cpp/

$ROOT = Split-Path $PSScriptRoot -Parent
$MODEL_PATH   = "$ROOT\backend\models\qwen3.5\Qwen3.5-4B-Q4_K_M.gguf"
$MMPROJ_PATH  = "$ROOT\backend\models\qwen3.5\mmproj-BF16.gguf"   # BF16 — быстрее F32
$LLAMA_SERVER = "$ROOT\llama.cpp\llama-server.exe"

# Проверяем наличие моделей
if (-not (Test-Path $MODEL_PATH)) {
    Write-Error "Модель не найдена: $MODEL_PATH"
    Write-Host "Запустите: python scripts\download_model.py"
    exit 1
}

if (-not (Test-Path $MMPROJ_PATH)) {
    Write-Error "mmproj-BF16.gguf не найден: $MMPROJ_PATH"
    Write-Host "Скачайте BF16 версию mmproj или переименуйте F32 → BF16"
    exit 1
}

if (-not (Test-Path $LLAMA_SERVER)) {
    Write-Error "llama-server.exe не найден: $LLAMA_SERVER"
    exit 1
}

Write-Host "Запуск OPTIMIZED llama-server (RTX 5070)" -ForegroundColor Green
Write-Host "Модель: $MODEL_PATH" -ForegroundColor Gray
Write-Host "mmproj: $MMPROJ_PATH" -ForegroundColor Gray
Write-Host "Порт: 8090" -ForegroundColor Gray
Write-Host ""

& $LLAMA_SERVER `
  --model $MODEL_PATH `
  --mmproj $MMPROJ_PATH `
  --port 8090 `
  --n-gpu-layers 99 `
  --mmproj-offload `
  --flash-attn on `
  --no-mmap `
  -c 12288 `
  -b 512 `
  -ub 512 `
  --image-min-tokens 1024 `
  --temp 0.1 `
  --top-p 0.9 `
  --top-k 50 `
  --presence-penalty 1.5 `
  --repeat-penalty 1.1 `
  --reasoning off `
  --no-webui
