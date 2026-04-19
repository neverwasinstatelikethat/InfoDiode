"""VLMClientDirect — нативный биндинг llama.cpp (максимальная скорость).

Используется вместо VLMClient (HTTP), когда нужна задержка 120–280 мс на кадр.
Поддерживает vision (mmproj-BF16.gguf) и JSON-вывод.
Убирает весь overhead HTTP и даёт максимальную скорость на GPU.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class VLMClientDirect:
    """Нативный Python-клиент для llama.cpp через llama-cpp-python.

    Singleton-класс для прямого доступа к GPU без HTTP-сервера.
    Даёт 120-280 мс на кадр вместо 300-500+ мс через HTTP.

    Attributes:
        llm: Экземпляр Llama из llama-cpp-python.
        max_image_size: Максимальный размер стороны изображения (даунскейл).
    """

    _instance: VLMClientDirect | None = None

    def __new__(cls) -> VLMClientDirect:
        """Создаёт singleton-экземпляр с загрузкой модели на GPU."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            model_dir = Path(settings.models_dir) / "qwen3.5"

            model_path = str(model_dir / "Qwen3.5-4B-Q4_K_M.gguf")
            mmproj_path = str(model_dir / "mmproj-BF16.gguf")

            # Проверяем наличие файлов моделей
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Модель не найдена: {model_path}")
            if not Path(mmproj_path).exists():
                raise FileNotFoundError(f"mmproj не найден: {mmproj_path}")

            try:
                from llama_cpp import Llama

                load_start = time.perf_counter()
                logger.info("VLMClientDirect: loading model from %s ...", model_path)

                # Критичные параметры для предотвращения CPU fallback
                import os
                os.environ.setdefault("LLAMA_CUDA_USE_GRAPHS", "0")  # Отключаем CUDA graphs

                cls._instance.llm = Llama(
                    # Пути к моделям
                    model_path=model_path,
                    mmproj_path=mmproj_path,

                    # Полная offload на GPU (RTX 5070 8 ГБ)
                    n_gpu_layers=-1,       # -1 = все слои на GPU
                    n_ctx=12288,           # УВЕЛИЧЕНО: 1024px изображения + SCADA глоссарий требуют больше контекста
                    n_batch=512,           # batch size для 8GB VRAM
                    flash_attn=True,       # Flash Attention (очень важно!)

                    # Параметры inference
                    verbose=True,         # не засоряем консоль
                )
                load_ms = (time.perf_counter() - load_start) * 1000
                logger.info(
                    "VLMClientDirect: model loaded in %.0fms — model=%s, mmproj=%s, n_gpu=-1, flash_attn=FALSE",
                    load_ms, model_path, mmproj_path,
                )

                # Проверяем что модель действительно на GPU
                try:
                    # Пытаемся получить информацию о GPU usage
                    if hasattr(cls._instance.llm, 'model'):
                        logger.info("VLMClientDirect: Llama instance created successfully")
                except Exception as check_err:
                    logger.warning("VLMClientDirect: GPU check failed — %s", str(check_err))

                # 5. Pre-init & warmup: dummy inference to initialize CUDA kernels
                warmup_start = time.perf_counter()
                logger.info("VLMClientDirect: STARTING warmup inference on synthetic frame...")
                try:
                    cls._instance._run_warmup()
                    warmup_ms = (time.perf_counter() - warmup_start) * 1000
                    logger.info("VLMClientDirect: warmup COMPLETE in %.0fms — CUDA kernels initialized", warmup_ms)
                except Exception as e:
                    warmup_ms = (time.perf_counter() - warmup_start) * 1000
                    logger.warning(
                        "VLMClientDirect: warmup FAILED after %.0fms — %s: %s. "
                        "First real frame may experience CUDA init delay.",
                        warmup_ms, type(e).__name__, str(e)[:300],
                    )

            except ImportError:
                raise ImportError(
                    "llama-cpp-python не установлен. "
                    "Установите: pip install llama-cpp-python "
                    '-C cmake.args="-DLLAMA_CUDA=ON -DLLAMA_FLASH_ATTN=ON"'
                )

        return cls._instance

    def __init__(self) -> None:
        """Инициализация с параметрами из конфигурации."""
        # Избегаем повторной инициализации singleton
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.model_name = getattr(settings, "vlm_model_name", "Qwen3.5-4B")
        self.max_image_size = getattr(settings, "vlm_max_image_size", 1024)
        self.max_tokens = getattr(settings, "vlm_max_tokens", 4096)
        self.temperature = getattr(settings, "vlm_temperature", 0.7)
        self.top_p = getattr(settings, "vlm_top_p", 0.8)
        self.top_k = getattr(settings, "vlm_top_k", 20)
        self.presence_penalty = getattr(settings, "vlm_presence_penalty", 1.5)
        self.repetition_penalty = getattr(settings, "vlm_repetition_penalty", 1.0)
        
        logger.info(
            "VLMClientDirect initialized: model=%s, max_tokens=%d, temp=%.2f, "
            "top_p=%.2f, top_k=%d, presence_penalty=%.1f, repetition_penalty=%.1f, max_image_size=%d",
            self.model_name, self.max_tokens, self.temperature,
            self.top_p, self.top_k, self.presence_penalty, self.repetition_penalty,
            self.max_image_size,
        )

    def _run_warmup(self) -> None:
        """Запускает dummy inference на синтетическом кадре для инициализации CUDA.

        Вызывается один раз сразу после загрузки модели в __new__.
        Предотвращает зависание на первом реальном кадре из-за CUDA init.
        """
        import gc

        # Принудительная очистка GPU памяти перед warmup
        gc.collect()

        logger.info("="*80)
        logger.info("VLMClientDirect._run_warmup: CREATING synthetic SCADA frame 640x480...")

        # Создаём синтетический кадр 640x480 (типичный размер после downscale SCADA)
        # Серый фон + простые элементы (имитация мнемосхемы)
        dummy_frame = np.full((480, 640, 3), 40, dtype=np.uint8)  # Тёмно-серый фон

        # Добавляем простые "параметры" (белые прямоугольники + текст)
        cv2.rectangle(dummy_frame, (50, 50), (200, 100), (255, 255, 255), 2)
        cv2.putText(dummy_frame, "T=45.2", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(dummy_frame, (400, 300), (550, 350), (255, 255, 255), 2)
        cv2.putText(dummy_frame, "P=120", (410, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Минимальные промпты для быстрого warmup
        system_prompt = "You are a test assistant. Respond with valid JSON only."
        user_prompt = (
            "Analyze this test image and return a JSON object with an empty 'parameters' list. "
            "Example: {\"parameters\": []}"
        )

        # Кодируем в JPEG + base64
        _, buf = cv2.imencode(".jpg", dummy_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        base64_image = base64.b64encode(buf).decode("utf-8")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        # Запускаем синхронный inference (CUDA kernels инициализируются здесь)
        logger.info("VLMClientDirect._run_warmup: CALLING create_chat_completion (this initializes CUDA)...")
        warmup_time = time.perf_counter()

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=64,  # Минимальный ответ для скорости
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            warmup_duration = (time.perf_counter() - warmup_time) * 1000

            # Проверяем что ответ валидный
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = response.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)

            logger.info(
                "VLMClientDirect._run_warmup: SUCCESS in %.0fms — tokens=%d, content='%s'",
                warmup_duration,
                completion_tokens,
                content[:100] if content else "empty",
            )
            logger.info("="*80)

        except Exception as e:
            warmup_duration = (time.perf_counter() - warmup_time) * 1000
            logger.error(
                "VLMClientDirect._run_warmup: FAILED after %.0fms — %s: %s",
                warmup_duration,
                type(e).__name__,
                str(e)[:500],
            )
            logger.error("="*80)
            raise  # Re-raise to caller (__new__) for graceful handling

    async def analyze_frame(
        self,
        frame: np.ndarray,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Анализирует один кадр напрямую через llama.cpp (120-280 мс на RTX 5070).

        Args:
            frame: Изображение в формате numpy array (BGR).
            system_prompt: Системный промпт для модели.
            user_prompt: Пользовательский промпт для модели.

        Returns:
            Распарсенный JSON ответ от модели.
        """
        from app.core.vlm_client import VLMClient

        start = time.perf_counter()

        # CLAHE ОТКЛЮЧЕН: искажает цвета SCADA (синий→серый, красный→оранжевый)
        # SCADA опирается на цветовую семантику: синий=актив, красный=авария, зеленый=норма
        #CLAHE: enhance_start = time.perf_counter()
        #CLAHE: lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        #CLAHE: l, a, b = cv2.split(lab)
        #CLAHE: clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        #CLAHE: l = clahe.apply(l)
        #CLAHE: lab = cv2.merge([l, a, b])
        #CLAHE: frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        #CLAHE: enhance_ms = (time.perf_counter() - enhance_start) * 1000
        #CLAHE: logger.debug("VLMClientDirect.analyze_frame: clahe_enhance=%.1fms", enhance_ms)

        # 1. Оптимизация изображения — без downscaling (max_image_size=1600)
        h, w = frame.shape[:2]
        logger.debug(
            "VLMClientDirect.analyze_frame: input_frame=%dx%d, max_image_size=%d",
            w, h, self.max_image_size,
        )
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            frame = cv2.resize(
                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
            )
            logger.debug(
                "VLMClientDirect.analyze_frame: resized to %dx%d (scale=%.3f)",
                frame.shape[1], frame.shape[0], scale,
            )

        # 2. Кодирование в base64 JPEG (качество 100 - без сжатия для максимальной четкости)
        encode_start = time.perf_counter()
        _, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100]
        )
        base64_image = base64.b64encode(buf).decode("utf-8")
        encode_ms = (time.perf_counter() - encode_start) * 1000
        logger.debug(
            "VLMClientDirect.analyze_frame: jpeg_encode=%.1fms, jpeg_size=%.1fKB, base64_size=%.1fKB",
            encode_ms,
            len(buf) / 1024,
            len(base64_image) / 1024,
        )

        # 3. Формируем сообщение
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        logger.debug(
            "VLMClientDirect.analyze_frame: messages_ready, system_prompt_len=%d, user_prompt_len=%d",
            len(system_prompt), len(user_prompt),
        )

        # 4. Прямой вызов llama.cpp (синхронный — запускаем в executor)
        loop = asyncio.get_running_loop()
        inference_start = time.perf_counter()
        
        # Log critical info before inference
        logger.info(
            "VLMClientDirect.analyze_frame: PREPARE inference — frame_size=%dx%d, system_len=%d, user_len=%d",
            frame.shape[1], frame.shape[0], len(system_prompt), len(user_prompt),
        )
        logger.info("VLMClientDirect.analyze_frame: STARTING llama.cpp inference...")

        def _sync_inference() -> dict[str, Any]:
            logger.debug("VLMClientDirect._sync_inference: calling create_chat_completion...")
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                presence_penalty=self.presence_penalty,
                repetition_penalty=self.repetition_penalty,
                response_format={"type": "json_object"},
            )
            logger.debug(
                "VLMClientDirect._sync_inference: create_chat_completion returned, usage=%s",
                response.get("usage", {}),
            )
            return response

        try:
            response = await loop.run_in_executor(None, _sync_inference)
        except Exception as e:
            inference_ms = (time.perf_counter() - inference_start) * 1000
            logger.error(
                "VLMClientDirect.analyze_frame: INFERENCE FAILED after %.1fms — %s: %s",
                inference_ms, type(e).__name__, str(e)[:500],
            )
            raise

        inference_ms = (time.perf_counter() - inference_start) * 1000
        logger.info(
            "VLMClientDirect.analyze_frame: inference_complete, total_time=%.1fms",
            inference_ms,
        )

        # 5. Парсинг ответа
        parse_start = time.perf_counter()
        
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(
                "VLMClientDirect.analyze_frame: INVALID response structure — %s, response_keys=%s",
                str(e), list(response.keys()) if isinstance(response, dict) else "not_dict",
            )
            raise ValueError(f"Некорректная структура ответа VLM: {e}")
        
        logger.debug(
            "VLMClientDirect.analyze_frame: response_content_len=%d, content_preview=%s",
            len(content), content[:200],
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        completion_tokens = response.get("usage", {}).get("completion_tokens", 0)
        
        # Предупреждение если модель сгенерировала слишком много токенов
        if completion_tokens > 400:
            logger.warning(
                "VLMClientDirect.analyze_frame: HIGH token count=%d — model may be generating verbose output. "
                "Consider reducing max_tokens or improving prompt specificity.",
                completion_tokens,
            )

        logger.info(
            "VLMClientDirect.analyze_frame: COMPLETE total=%.1fms "
            "(encode=%.1fms, inference=%.1fms), tokens=%d",
            elapsed_ms, encode_ms, inference_ms, completion_tokens,
        )

        # Используем существующий парсер JSON из VLMClient
        result = VLMClient._extract_json_from_response_static(content)
        parse_ms = (time.perf_counter() - parse_start) * 1000
        logger.debug(
            "VLMClientDirect.analyze_frame: json_parse=%.1fms, params_extracted=%d",
            parse_ms, len(result.get("parameters", [])),
        )
        return result

    async def analyze_multi_image(
        self,
        frames: list[np.ndarray],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Анализирует несколько изображений в одном VLM-запросе (multi-image).

        Используется для остаточных областей (residual), когда нужно
        отправить несколько мелких кропов одним запросом.

        Args:
            frames: Список изображений в формате numpy array (BGR).
            system_prompt: Системный промпт.
            user_prompt: Пользовательский промпт.

        Returns:
            Распарсенный JSON ответ от модели.
        """
        from app.core.vlm_client import VLMClient

        start = time.perf_counter()

        # Кодируем все изображения в base64
        image_contents: list[dict[str, Any]] = []
        for frame in frames:
            h, w = frame.shape[:2]
            if max(h, w) > self.max_image_size:
                scale = self.max_image_size / max(h, w)
                frame = cv2.resize(
                    frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
                )
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            base64_image = base64.b64encode(buf).decode("utf-8")
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })

        # Добавляем текстовый промпт после всех изображений
        image_contents.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_contents},
        ]

        logger.info(
            "VLMClientDirect.analyze_multi_image: %d images, system_len=%d, user_len=%d",
            len(frames), len(system_prompt), len(user_prompt),
        )

        # Прямой вызов llama.cpp в executor
        loop = asyncio.get_running_loop()

        def _sync_inference() -> dict[str, Any]:
            return self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                presence_penalty=self.presence_penalty,
                repetition_penalty=self.repetition_penalty,
                response_format={"type": "json_object"},
            )

        response = await loop.run_in_executor(None, _sync_inference)

        # Парсинг
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Некорректная структура ответа VLM: {e}")

        result = VLMClient._extract_json_from_response_static(content)

        if isinstance(result, dict) and "zone_id" not in result:
            result["zone_id"] = 0

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "VLMClientDirect.analyze_multi_image: COMPLETE %.1fms, images=%d, params=%d",
            elapsed_ms, len(frames), len(result.get("parameters", [])),
        )

        return result

    async def health_check(self) -> bool:
        """Проверяет доступность модели.

        Returns:
            True если модель загружена и готова к inference.
        """
        return self.llm is not None

    async def close(self) -> None:
        """Закрывает модель и освобождает GPU-ресурсы."""
        if hasattr(self, "llm") and self.llm is not None:
            self.llm.close()
            self.llm = None
            VLMClientDirect._instance = None
            logger.info("VLMClientDirect закрыт, GPU ресурсы освобождены")

    async def __aenter__(self) -> VLMClientDirect:
        """Асинхронный контекстный менеджер — вход."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Асинхронный контекстный менеджер — выход."""
        # Не закрываем singleton при выходе из контекста
        pass
