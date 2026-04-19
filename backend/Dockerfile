FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем зависимости
COPY pyproject.toml ./

# Устанавливаем Python-пакеты
RUN python3.12 -m pip install --no-cache-dir --upgrade pip && \
    python3.12 -m pip install --no-cache-dir .

# Предзагрузка моделей PaddleOCR (офлайн-режим)
RUN python3.12 -c "\
from paddleocr import PaddleOCR; \
ocr = PaddleOCR(use_angle_cls=True, lang='ru', use_gpu=False, show_log=True, \
    det_model_dir='/app/models/paddleocr/det', \
    rec_model_dir='/app/models/paddleocr/rec', \
    cls_model_dir='/app/models/paddleocr/cls'); \
print('Models downloaded successfully')"

# Копируем исходный код
COPY app/ ./app/

# Директории данных
RUN mkdir -p /app/data/input_videos /app/data/output_xml /app/data/qr_codes /app/data/encryption_keys /app/models/paddleocr

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python3.12", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
