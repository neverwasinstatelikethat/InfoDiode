"""Florence-2 Region Proposer — семантический OCR path.

VLM (Vision Language Model) идеально подходит для SCADA-мнемосхем:
- Не нужно дообучение — работает с любыми мнемосхемами
- OCR_WITH_REGION даёт текст + quad для каждого региона
- Семантически осмысленные регионы (связные информационные блоки)
- Запускается раз в N секунд (кэшируется), Paddle — каждый кадр

Использует Florence-2-large (pretrained, 4096 token context) для максимального качества
и стабильности на больших SCADA-кадрах. Florence-2-large-ft имеет ограничение positional
embeddings, вызывающее IndexError на изображениях >768px.
Оптимизации: float16, GPU auto-detect, num_beams=3, глобальный singleton.

На основе официальной документации:
https://huggingface.co/microsoft/Florence-2-large

Модель поддерживает таски:
- <OCR> — только текст (без bbox), быстрый
- <OCR_WITH_REGION> — текст + quad_box для каждого региона
- <OD> — object detection (bboxes + labels)
- <DENSE_REGION_CAPTION> — плотные описания регионов

Confidence scores:
- Florence-2-large поддерживает confidence через compute_transition_scores()
  (return_dict_in_generate=True, output_scores=True)
- Florence-2-large-ft пока не имеет встроенной поддержки confidence в post_process_generation
- Мы используем num_beams=3 и возвращаем средний beam score как confidence

Требования: transformers>=4.41.0, torch
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np

from app.core.ocr_models import BBox, SourceType, TextBox

logger = logging.getLogger(__name__)

# Глобальный singleton модели (ленивая загрузка)
_florence_model = None
_florence_processor = None
_florence_device = "cpu"
_florence_dtype = None  # torch.dtype, определяется при загрузке

# Модель по умолчанию: large (pretrained) — поддерживает 4096 токенов, нет проблем с embed_positions
# Florence-2-large-ft (finetuned) имеет ограничение positional embeddings ~768px,
# что вызывает IndexError на больших SCADA-кадрах
DEFAULT_MODEL_ID = "microsoft/Florence-2-large"

# Альтернатива: Florence-2-large-ft (finetuned, 2048 token context)
# Лучше для некоторых downstream задач, но ограничение positional embeddings
FT_MODEL_ID = "microsoft/Florence-2-large-ft"


def _detect_device() -> str:
    """Определяет лучшее устройство для инференса.
    
    Приоритет: CUDA GPU → CPU
    Проверяет доступность torch.cuda и выбирает оптимальное устройство.
    
    Returns:
        "cuda:0" если GPU доступен, иначе "cpu".
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info("GPU обнаружен: %s (%.1f GB VRAM) — будет использоваться для Florence-2", name, vram)
            return "cuda:0"
        else:
            logger.info("GPU не доступен (torch.cuda.is_available()=False) — используется CPU")
    except ImportError:
        logger.info("torch не установлен — используется CPU")
    return "cpu"


def _load_florence(
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
) -> tuple:
    """Ленивая загрузка Florence-2 модели (singleton).

    Согласно официальной документации:
    - Все модели обучены в float16
    - Используем AutoModelForCausalLM + AutoProcessor
    - trust_remote_code=True обязателен

    Returns:
        (model, processor) tuple.
    """
    global _florence_model, _florence_processor, _florence_device, _florence_dtype

    if _florence_model is not None and _florence_device == device:
        return _florence_model, _florence_processor

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError:
        raise ImportError(
            "transformers/torch не установлен. Установите: pip install transformers torch"
        )

    # Автоопределение устройства
    if device is None:
        device = _detect_device()

    logger.info("Загрузка Florence-2 (%s) на %s...", model_id, device)
    t0 = time.perf_counter()

    # Проверяем локальный кэш модели
    local_dir = Path(__file__).resolve().parent.parent.parent / "models" / "florence2"
    model_path = str(local_dir) if local_dir.exists() else model_id

    # Все Florence-2 модели обучены в float16 — используем его на GPU
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model.eval()
    model = model.to(device, torch_dtype)

    _florence_model = model
    _florence_processor = processor
    _florence_device = device
    _florence_dtype = torch_dtype

    elapsed = time.perf_counter() - t0
    logger.info(
        "Florence-2 загружена: %s на %s (%s) за %.1fs",
        model_id, device, torch_dtype, elapsed,
    )
    return model, processor


class FlorenceDetector:
    """Florence-2 OCR_WITH_REGION — семантический path.

    Использует Florence-2-large (pretrained) модель для максимального
    качества распознавания текста на SCADA-мнемосхемах.

    Запускается раз в N секунд. Даёт семантически осмысленные регионы
    HMI-экрана — не просто текст, а связные информационные блоки.

    Согласно документации:
    - OCR_WITH_REGION возвращает quad_boxes + labels
    - OCR возвращает только текст (без bbox) — быстрее
    - num_beams=3 — стандартное значение из документации
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        num_beams: int | None = None,
    ) -> None:
        self._model_id = model_id
        self._device = device  # None = auto-detect
        self._model = None
        self._processor = None
        self._confidence_available: bool | None = None  # determined on first call
        # num_beams из конфигурации или переданный явно
        if num_beams is not None:
            self._num_beams = num_beams
        else:
            from app.config import settings
            self._num_beams = settings.florence_num_beams

    def _ensure_loaded(self) -> None:
        """Ленивая загрузка модели."""
        if self._model is None:
            t_load = time.perf_counter()
            self._model, self._processor = _load_florence(
                self._model_id, self._device
            )
            load_ms = (time.perf_counter() - t_load) * 1000
            logger.info("Florence-2 модель загружена за %.0fms", load_ms)

    def propose(self, frame: np.ndarray) -> list[TextBox]:
        """Запускает Florence-2 OCR_WITH_REGION на кадре.

        Согласно официальной документации:
        - Передаём input_ids и pixel_values явно
        - .to(device, torch_dtype) для корректной работы float16
        - num_beams=1 (greedy decoding для скорости, 3x быстрее чем beam search)
        - max_new_tokens=1024 (SCADA-кадру достаточно ~800 токенов)

        Args:
            frame: Входной кадр BGR.

        Returns:
            Список TextBox с текстом, координатами и confidence.
        """
        self._ensure_loaded()

        try:
            from PIL import Image
            import torch
        except ImportError:
            logger.warning("PIL/torch не доступны — Florence пропущена")
            return []

        t0 = time.perf_counter()

        # BGR → RGB: создаёт копию, но необходимо для корректности цветов
        # PIL ожидает RGB, OpenCV даёт BGR
        t1 = time.perf_counter()
        pil = Image.fromarray(frame[..., ::-1])
        pil_conv_ms = (time.perf_counter() - t1) * 1000

        # Florence-2-large (pretrained) поддерживает изображения до ~1536px
        # (4096 token context). Florence-2-large-ft ограничена ~768px (2048 tokens).
        # При превышении embed_positions выходит за пределы: IndexError
        max_florence_dim = 1536
        w, h = pil.size
        if max(w, h) > max_florence_dim:
            t_resize = time.perf_counter()
            scale = max_florence_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil = pil.resize((new_w, new_h), Image.LANCZOS)
            resize_ms = (time.perf_counter() - t_resize) * 1000
            logger.debug("Florence: кадр уменьшен %dx%d -> %dx%d (%.0fms)", w, h, new_w, new_h, resize_ms)

        task = "<OCR_WITH_REGION>"

        # Согласно документации: processor(text=prompt, images=image)
        t2 = time.perf_counter()
        inputs = self._processor(
            text=task, images=pil, return_tensors="pt"
        )
        processor_ms = (time.perf_counter() - t2) * 1000

        # Согласно документации: .to(device, torch_dtype)
        device = self._device or _florence_device
        dtype = _florence_dtype or torch.float32
        
        # Логируем устройство при первом вызове для верификации
        if not hasattr(self, '_device_logged'):
            import torch
            is_using_gpu = device.startswith("cuda")
            gpu_active = torch.cuda.is_available() if device.startswith("cuda") else False
            if is_using_gpu and gpu_active:
                vram_used = torch.cuda.memory_allocated(device) / (1024**2)
                logger.info(
                    "Florence-2 использует GPU: %s (VRAM: %.1f MB, dtype: %s)",
                    device, vram_used, dtype
                )
            else:
                logger.warning(
                    "Florence-2 использует CPU (device=%s, GPU available=%s) — это будет медленно!",
                    device, torch.cuda.is_available()
                )
            self._device_logged = True
        
        t3 = time.perf_counter()
        inputs = inputs.to(device, dtype)
        to_device_ms = (time.perf_counter() - t3) * 1000

        # Прямой generate без confidence scores (быстрее, надёжнее)
        # max_new_tokens=1024: SCADA-кадр содержит ~50-100 текстовых полей,
        # что занимает ~500-800 токенов. 4096 токенов избыточно и замедляет
        # генерацию в 4-8 раз (особенно при num_beams > 1).
        # Примечание: early_stopping не используется — num_beams=1 (greedy decoding)
        generate_kwargs: dict = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "max_new_tokens": 1024,  # SCADA-кадру достаточно 1024 токенов
            "num_beams": self._num_beams,
            "do_sample": False,
        }
        try:
            t4 = time.perf_counter()
            generated_ids = self._model.generate(**generate_kwargs)
            generate_ms = (time.perf_counter() - t4) * 1000
        except (IndexError, RuntimeError) as e:
            # IndexError: embed_positions — слишком большое изображение
            # RuntimeError: CUDA OOM и другие GPU ошибки
            logger.warning("Florence generate error: %s — пропускаем кадр", e)
            return []
        t5 = time.perf_counter()
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        decode_ms = (time.perf_counter() - t5) * 1000
        
        t6 = time.perf_counter()
        parsed = self._processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(pil.width, pil.height),
        )
        postprocess_ms = (time.perf_counter() - t6) * 1000

        # parsed = {"<OCR_WITH_REGION>": {"quad_boxes": [...], "labels": [...]}}
        data = parsed.get("<OCR_WITH_REGION>", {})
        boxes: list[TextBox] = []
        quads = data.get("quad_boxes", [])
        labels = data.get("labels", [])
        official_scores = data.get("scores", None)  # Официальные скоры если доступны

        for i, (quad, text) in enumerate(zip(quads, labels)):
            try:
                quad_arr = np.array(quad, dtype=np.float32).reshape(4, 2)
                bbox = BBox.from_quad(quad_arr)
            except (ValueError, TypeError) as e:
                logger.debug("Невалидный quad для текста '%s': %s", text, e)
                continue

            text_clean = text.strip() if isinstance(text, str) else str(text).strip()
            if not text_clean:
                continue

            # Используем официальный скор если доступен, иначе эвристику
            if official_scores is not None and i < len(official_scores):
                conf = float(official_scores[i])
            else:
                conf = self._calculate_confidence(text_clean)

            boxes.append(
                TextBox(
                    bbox=bbox,
                    text=text_clean,
                    confidence=round(conf, 4),
                    source="florence",
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        # Добавляем информацию о VRAM если используется GPU
        extra_info = ""
        if device.startswith("cuda"):
            try:
                import torch
                vram_allocated = torch.cuda.memory_allocated(device) / (1024**2)
                vram_reserved = torch.cuda.memory_reserved(device) / (1024**2)
                extra_info = f", VRAM: {vram_allocated:.0f}/{vram_reserved:.0f} MB"
            except Exception:
                pass
        
        logger.info(
            "Florence-2: %d регионов за %.0fms [BGR→RGB:%.0fms, processor:%.0fms, to_device:%.0fms, generate:%.0fms, decode:%.0fms, postprocess:%.0fms] (device=%s%s, num_beams=%d)",
            len(boxes), elapsed_ms, pil_conv_ms, processor_ms, to_device_ms, generate_ms, decode_ms, postprocess_ms, device, extra_info, self._num_beams,
        )
        return boxes

    def _calculate_confidence(self, text: str) -> float:
        """Эвристика confidence для Florence (нет нативных скоров)."""
        if not text:
            return 0.5
        base = 0.70 + min(len(text) * 0.01, 0.15)
        unique_chars = len(set(text))
        diversity_ratio = unique_chars / max(len(text), 1)
        diversity_penalty = (1.0 - diversity_ratio) * 0.1
        return max(0.55, min(0.85, base - diversity_penalty))

    def _run_with_confidence(
        self,
        frame: np.ndarray,
        task: str,
        text_input: str = "",
    ) -> tuple[dict, float]:
        """Выполняет инференс Florence-2 с вычислением confidence score.

        Универсальный helper для всех multi-task запросов. Выполняет resize,
        generate с return_dict_in_generate=True для получения beam scores,
        и возвращает распарсенный результат вместе с confidence.

        Args:
            frame: Входной кадр BGR.
            task: Таск-токен Florence-2 (например, "<DETAILED_CAPTION>").
            text_input: Дополнительный текст для таски (например, caption для grounding).

        Returns:
            Кортеж (parsed_result, confidence), где parsed_result — словарь
            из post_process_generation, confidence — средний beam score [0, 1].
        """
        self._ensure_loaded()

        try:
            from PIL import Image
            import torch
        except ImportError:
            logger.warning("PIL/torch не доступны — инференс пропущен")
            return {}, 0.0

        t0 = time.perf_counter()

        # BGR → RGB
        pil = Image.fromarray(frame[..., ::-1])

        # Resize для Florence-2 (max 1536px — см. propose())
        max_florence_dim = 1536
        orig_w, orig_h = pil.size
        scale = 1.0
        if max(orig_w, orig_h) > max_florence_dim:
            scale = max_florence_dim / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            pil = pil.resize((new_w, new_h), Image.LANCZOS)
            logger.debug(
                "Florence _run_with_confidence: кадр уменьшен %dx%d -> %dx%d",
                orig_w, orig_h, new_w, new_h,
            )

        # Формируем полный prompt
        prompt = task + text_input if text_input else task

        # Processor
        inputs = self._processor(
            text=prompt, images=pil, return_tensors="pt"
        )
        device = self._device or _florence_device
        dtype = _florence_dtype or torch.float32
        inputs = inputs.to(device, dtype)

        # Generate с return_dict_in_generate=True для confidence scores
        # num_beams=1 для greedy decoding — 3x быстрее чем beam search
        generate_kwargs: dict = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "max_new_tokens": 1024,
            "num_beams": 1,  # Greedy decoding для скорости (3x быстрее)
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        try:
            generated_ids = self._model.generate(**generate_kwargs)
        except (IndexError, RuntimeError) as e:
            logger.warning("Florence generate error: %s — пропускаем", e)
            return {}, 0.0

        # Вычисляем confidence из scores напрямую (num_beams=1)
        # С num_beams=1 compute_transition_scores не работает, используем scores напрямую
        try:
            if generated_ids.scores:
                # scores — tuple of tensors, каждый shape (batch_size, vocab_size)
                # Берём log_softmax max prob для каждого шага как confidence
                step_confs = []
                for step_scores in generated_ids.scores:
                    probs = torch.nn.functional.softmax(step_scores[0], dim=-1)
                    max_prob = probs.max().item()
                    step_confs.append(max_prob)
                confidence = sum(step_confs) / len(step_confs) if step_confs else 0.5
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 0.5
        except Exception as e:
            logger.debug("Не удалось вычислить confidence из scores: %s", e)
            confidence = 0.5

        # Декодируем и парсим результат
        generated_text = self._processor.batch_decode(
            generated_ids.sequences, skip_special_tokens=False
        )[0]
        parsed = self._processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(pil.width, pil.height),
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Florence-2 %s: confidence=%.3f за %.0fms",
            task, confidence, elapsed_ms,
        )

        return parsed, confidence, scale


    def ocr_only(self, frame: np.ndarray) -> str:
        """Запускает Florence-2 <OCR> (только текст, без bbox).

        Быстрее чем OCR_WITH_REGION — не генерирует координаты.
        Полезно как дополнение для проверки распознанного текста.

        Оптимизации:
        - max_new_tokens=1024 (SCADA-кадру достаточно ~800 токенов)
        - num_beams=1 (greedy decoding, 3x быстрее чем beam search)

        Args:
            frame: Входной кадр BGR.

        Returns:
            Распознанный текст (многострочный).
        """
        self._ensure_loaded()

        try:
            from PIL import Image
            import torch
        except ImportError:
            return ""

        # BGR → RGB: создаёт копию, но необходимо для корректности цветов
        pil = Image.fromarray(frame[..., ::-1])

        # Ограничение размера для Florence-2 (та же причина — embed_positions)
        max_florence_dim = 1536
        w, h = pil.size
        scale = 1.0
        if max(w, h) > max_florence_dim:
            scale = max_florence_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil = pil.resize((new_w, new_h), Image.LANCZOS)

        task = "<OCR>"
        inputs = self._processor(
            text=task, images=pil, return_tensors="pt"
        )
        device = self._device or _florence_device
        dtype = _florence_dtype or torch.float32
        inputs = inputs.to(device, dtype)

        # max_new_tokens=1024: SCADA-кадру достаточно 1024 токенов (см. propose())
        # Примечание: early_stopping не используется — num_beams=1 (greedy decoding)
        generate_kwargs: dict = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "max_new_tokens": 1024,  # SCADA-кадру достаточно 1024 токенов
            "num_beams": self._num_beams,
            "do_sample": False,
        }
        try:
            generated_ids = self._model.generate(**generate_kwargs)
        except (IndexError, RuntimeError) as e:
            logger.warning("Florence generate error: %s — пропускаем кадр", e)
            return ""
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = self._processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(pil.width, pil.height),
        )

        # parsed = {"<OCR>": "recognized text..."}
        return parsed.get("<OCR>", "")

    def ground_phrases(
        self, frame: np.ndarray, phrases: list[str]
    ) -> list[tuple[str, BBox | None, float]]:
        """Выполняет phrase grounding для списка фраз на кадре.

        Использует Florence-2 task <CAPTION_TO_PHRASE_GROUNDING> для локализации
        параметров на SCADA-мнемосхеме. Строит caption из имён параметров,
        разделённых точкой, и возвращает bbox для каждой найденной фразы.

        Args:
            frame: Входной кадр BGR.
            phrases: Список фраз для поиска (например, названия параметров).

        Returns:
            Список кортежей (phrase, BBox_or_None, confidence) — None если фраза
            не найдена. Координаты bbox соответствуют исходному размеру кадра.
        """
        self._ensure_loaded()

        try:
            from PIL import Image
            import torch
        except ImportError:
            logger.warning("PIL/torch не доступны — Florence grounding пропущен")
            return [(phrase, None, 0.0) for phrase in phrases]

        if not phrases:
            return []

        # Florence-2 имеет ограничение контекста — разбиваем на батчи по 20 фраз
        max_phrases_per_call = 20
        results: list[tuple[str, BBox | None, float]] = []

        for batch_start in range(0, len(phrases), max_phrases_per_call):
            batch_phrases = phrases[batch_start:batch_start + max_phrases_per_call]
            caption = " . ".join(batch_phrases)

            try:
                parsed, confidence, scale = self._run_with_confidence(
                    frame, "<CAPTION_TO_PHRASE_GROUNDING>", caption
                )
            except (IndexError, RuntimeError) as e:
                logger.warning(
                    "Florence grounding error для батча %d-%d: %s",
                    batch_start, batch_start + len(batch_phrases), e,
                )
                results.extend([(phrase, None, 0.0) for phrase in batch_phrases])
                continue

            # parsed = {"<CAPTION_TO_PHRASE_GROUNDING>": {"bboxes": [...], "labels": [...]}}
            data = parsed.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
            bboxes = data.get("bboxes", [])
            labels = data.get("labels", [])

            # Создаём словарь найденных фраз -> bbox
            found_bboxes: dict[str, BBox] = {}
            for bbox_coords, label in zip(bboxes, labels):
                try:
                    x1, y1, x2, y2 = bbox_coords
                    # Масштабируем обратно к исходному размеру кадра
                    if scale < 1.0:
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)
                    bbox = BBox(
                        x=int(min(x1, x2)),
                        y=int(min(y1, y2)),
                        w=int(abs(x2 - x1)),
                        h=int(abs(y2 - y1)),
                    )
                    found_bboxes[label] = bbox
                except (ValueError, TypeError) as e:
                    logger.debug("Невалидный bbox для фразы '%s': %s", label, e)
                    continue

            # Для каждой фразы в батче определяем результат
            for phrase in batch_phrases:
                if phrase in found_bboxes:
                    results.append((phrase, found_bboxes[phrase], confidence))
                else:
                    results.append((phrase, None, confidence))

        logger.info(
            "Florence grounding: найдено %d/%d фраз",
            sum(1 for _, bbox, _ in results if bbox is not None), len(phrases),
        )
        return results

    def ocr_all_text(self, frame: np.ndarray) -> list[tuple[str, BBox, float]]:
        """Распознаёт весь текст на кадре с координатами и confidence.

        Использует <OCR_WITH_REGION> напрямую через _run_with_confidence
        для получения confidence scores для каждого распознанного региона.

        Args:
            frame: Входной кадр BGR.

        Returns:
            Список кортежей (text, BBox, confidence) для каждого региона.
        """
        try:
            parsed, overall_confidence, scale = self._run_with_confidence(
                frame, "<OCR_WITH_REGION>"
            )
        except (IndexError, RuntimeError) as e:
            logger.warning("Florence OCR_WITH_REGION error: %s", e)
            return []

        # parsed = {"<OCR_WITH_REGION>": {"quad_boxes": [...], "labels": [...]}}
        data = parsed.get("<OCR_WITH_REGION>", {})
        quads = data.get("quad_boxes", [])
        labels = data.get("labels", [])
        scores = data.get("scores", None)  # Официальные скоры если доступны

        results: list[tuple[str, BBox, float]] = []
        for i, (quad, text) in enumerate(zip(quads, labels)):
            try:
                quad_arr = np.array(quad, dtype=np.float32).reshape(4, 2)
                # Масштабируем обратно к исходному размеру кадра
                if scale < 1.0:
                    quad_arr = quad_arr / scale
                bbox = BBox.from_quad(quad_arr)
            except (ValueError, TypeError) as e:
                logger.debug("Невалидный quad для текста '%s': %s", text, e)
                continue

            text_clean = text.strip() if isinstance(text, str) else str(text).strip()
            if not text_clean:
                continue

            # Используем официальный скор если доступен, иначе overall confidence
            if scores is not None and i < len(scores):
                conf = float(scores[i])
            else:
                conf = overall_confidence

            results.append((text_clean, bbox, round(conf, 4)))

        logger.info("Florence OCR_WITH_REGION: %d регионов", len(results))
        return results

    def describe_scene(self, frame: np.ndarray) -> dict:
        """Описывает сцену на SCADA-мнемосхеме.

        Использует <DETAILED_CAPTION> для получения детального описания
        изображения. Парсит описание для извлечения структурных элементов:
        тип ГПА, активная вкладка, наличие popup, имя мнемосхемы.

        Args:
            frame: Входной кадр BGR.

        Returns:
            Словарь с ключами:
            - description: полное текстовое описание
            - mnemonic_name: имя мнемосхемы (если найдено)
            - has_popup: bool, есть ли модальное окно/popup
            - active_tab: активная вкладка (если определена)
            - gpa_type: тип ГПА (например, "GPA-21")
        """
        try:
            parsed, confidence, _ = self._run_with_confidence(
                frame, "<DETAILED_CAPTION>"
            )
        except (IndexError, RuntimeError) as e:
            logger.warning("Florence DETAILED_CAPTION error: %s", e)
            return {
                "description": "",
                "mnemonic_name": None,
                "has_popup": False,
                "active_tab": None,
                "gpa_type": None,
            }

        # parsed = {"<DETAILED_CAPTION>": "описание текст..."}
        description = parsed.get("<DETAILED_CAPTION>", "")

        # Парсим описание для извлечения структурных элементов
        result = {
            "description": description,
            "mnemonic_name": None,
            "has_popup": False,
            "active_tab": None,
            "gpa_type": None,
        }

        if not description:
            return result

        # Извлекаем тип ГПА
        # Прямое совпадение: ГПА-XX или GPA-XX
        gpa_match = re.search(r"(ГПА|GPA)[-\s]?(\d+)", description, re.IGNORECASE)
        if gpa_match:
            result["gpa_type"] = f"GPA-{gpa_match.group(2)}"
        else:
            # Эвристики по ключевым словам
            if "Запал" in description or "запал" in description:
                result["gpa_type"] = "GPA-11"
            elif ("КВД" in description or "квд" in description) and (
                "КС" in description or "кс" in description
            ):
                result["gpa_type"] = "GPA-21"

        # Определяем активную вкладку
        tab_keywords = {
            "К→М": ["К→М", "К-М", "К в М"],
            "МГ→К": ["МГ→К", "МГ-К", "МГ в К"],
            "АП": [" АП ", " АП,", " АП.", "вкладка АП"],
            "ГР без газа": ["ГР без газа", "ГР-без газа"],
            "САУ": [" САУ ", " САУ,", "вкладка САУ"],
            "Тренд": ["Тренд", "тренд"],
        }
        for tab_name, patterns in tab_keywords.items():
            for pattern in patterns:
                if pattern in description:
                    result["active_tab"] = tab_name
                    break
            if result["active_tab"]:
                break

        # Определяем наличие popup/модального окна
        popup_keywords = [
            "диалог",
            "modal",
            "popup",
            "всплывающее",
            "окно",
            "диалоговое",
        ]
        result["has_popup"] = any(kw in description.lower() for kw in popup_keywords)

        # Извлекаем имя мнемосхемы (обычно в начале описания)
        # Паттерн: "мнемосхема X" или "схема X"
        mnemonic_match = re.search(
            r"(мнемосхема|схема|экран)\s+([\w\-\s]+?)(?:\.|,|$)",
            description,
            re.IGNORECASE,
        )
        if mnemonic_match:
            result["mnemonic_name"] = mnemonic_match.group(2).strip()

        logger.info(
            "Florence describe_scene: GPA=%s, tab=%s, popup=%s",
            result["gpa_type"], result["active_tab"], result["has_popup"],
        )
        return result

    def detect_regions(self, frame: np.ndarray) -> list[tuple[str, BBox]]:
        """Детектирует регионы с плотными описаниями.

        Использует <DENSE_REGION_CAPTION> для получения описаний регионов
        с их bounding boxes. Полезно для понимания структуры мнемосхемы.

        Args:
            frame: Входной кадр BGR.

        Returns:
            Список кортежей (caption, BBox) для каждого детектированного региона.
        """
        try:
            parsed, confidence, scale = self._run_with_confidence(
                frame, "<DENSE_REGION_CAPTION>"
            )
        except (IndexError, RuntimeError) as e:
            logger.warning("Florence DENSE_REGION_CAPTION error: %s", e)
            return []

        # parsed = {"<DENSE_REGION_CAPTION>": {"bboxes": [...], "labels": [...]}}
        data = parsed.get("<DENSE_REGION_CAPTION>", {})
        bboxes = data.get("bboxes", [])
        labels = data.get("labels", [])

        results: list[tuple[str, BBox]] = []
        for bbox_coords, caption in zip(bboxes, labels):
            try:
                x1, y1, x2, y2 = bbox_coords
                # Масштабируем обратно к исходному размеру кадра
                if scale < 1.0:
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                bbox = BBox(
                    x=int(min(x1, x2)),
                    y=int(min(y1, y2)),
                    w=int(abs(x2 - x1)),
                    h=int(abs(y2 - y1)),
                )
                results.append((caption, bbox))
            except (ValueError, TypeError) as e:
                logger.debug("Невалидный bbox для региона '%s': %s", caption, e)
                continue

        logger.info("Florence detect_regions: %d регионов", len(results))
        return results

    def detect_objects(self, frame: np.ndarray) -> list[tuple[str, BBox, float]]:
        """Детектирует объекты на кадре.

        Использует <OD> (Object Detection) для обнаружения объектов
        с их классами, bounding boxes и confidence scores.

        Args:
            frame: Входной кадр BGR.

        Returns:
            Список кортежей (label, BBox, confidence) для каждого объекта.
        """
        try:
            parsed, overall_confidence, scale = self._run_with_confidence(
                frame, "<OD>"
            )
        except (IndexError, RuntimeError) as e:
            logger.warning("Florence OD error: %s", e)
            return []

        # parsed = {"<OD>": {"bboxes": [...], "labels": [...]}}
        data = parsed.get("<OD>", {})
        bboxes = data.get("bboxes", [])
        labels = data.get("labels", [])
        scores = data.get("scores", None)  # Официальные скоры если доступны

        results: list[tuple[str, BBox, float]] = []
        for i, (bbox_coords, label) in enumerate(zip(bboxes, labels)):
            try:
                x1, y1, x2, y2 = bbox_coords
                # Масштабируем обратно к исходному размеру кадра
                if scale < 1.0:
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                bbox = BBox(
                    x=int(min(x1, x2)),
                    y=int(min(y1, y2)),
                    w=int(abs(x2 - x1)),
                    h=int(abs(y2 - y1)),
                )
                # Используем официальный скор если доступен
                conf = float(scores[i]) if scores and i < len(scores) else overall_confidence
                results.append((label, bbox, round(conf, 4)))
            except (ValueError, TypeError) as e:
                logger.debug("Невалидный bbox для объекта '%s': %s", label, e)
                continue

        logger.info("Florence detect_objects: %d объектов", len(results))
        return results

    def propose_regions(self, frame: np.ndarray) -> list[BBox]:
        """Предлагает регионы интереса без классификации.

        Использует <REGION_PROPOSAL> для получения bounding boxes
        потенциально интересных регионов. Не возвращает классы/метки.

        Args:
            frame: Входной кадр BGR.

        Returns:
            Список BBox для каждого предложенного региона.
        """
        try:
            parsed, confidence, scale = self._run_with_confidence(
                frame, "<REGION_PROPOSAL>"
            )
        except (IndexError, RuntimeError) as e:
            logger.warning("Florence REGION_PROPOSAL error: %s", e)
            return []

        # parsed = {"<REGION_PROPOSAL>": {"bboxes": [...], "labels": [...]}}
        data = parsed.get("<REGION_PROPOSAL>", {})
        bboxes = data.get("bboxes", [])

        results: list[BBox] = []
        for bbox_coords in bboxes:
            try:
                x1, y1, x2, y2 = bbox_coords
                # Масштабируем обратно к исходному размеру кадра
                if scale < 1.0:
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                bbox = BBox(
                    x=int(min(x1, x2)),
                    y=int(min(y1, y2)),
                    w=int(abs(x2 - x1)),
                    h=int(abs(y2 - y1)),
                )
                results.append(bbox)
            except (ValueError, TypeError) as e:
                logger.debug("Невалидный bbox региона: %s", e)
                continue

        logger.info("Florence propose_regions: %d регионов", len(results))
        return results

    @staticmethod
    def is_available() -> bool:
        """Проверяет доступность Florence-2."""
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM  # noqa: F401
            return True
        except ImportError:
            return False


def warmup_florence(device: str | None = None) -> float:
    """Прогревает Florence-2 модель для JIT-компиляции CUDA-ядер.

    Загружает модель (если ещё не загружена) и запускает инференс
    на тестовых изображениях для инициализации CUDA-ядер и
    предварительного выделения GPU-памяти.
    Использует размер изображения, близкий к реальному SCADA-кадру (1280x768),
    чтобы CUDA выделила память нужного размера с первого раза.
    Вызывается при старте приложения.

    Args:
        device: Устройство для инференса (None = автоопределение).

    Returns:
        Время прогрева в секундах (0.0 если модель недоступна).
    """
    if not FlorenceDetector.is_available():
        logger.info("Florence-2 недоступна — прогрев пропущен")
        return 0.0

    logger.info("Прогрев Florence-2 (JIT-компиляция CUDA-ядер + GPU memory allocation)...")
    t0 = time.perf_counter()

    try:
        # Загружаем модель если ещё не загружена
        model, processor = _load_florence(device=device)

        from PIL import Image
        import torch

        actual_device = device or _florence_device
        dtype = _florence_dtype or torch.float32

        # Прогрев 1: маленькое изображение для быстрой инициализации
        dummy_small = Image.new("RGB", (100, 100), color="black")
        task = "<OCR>"
        inputs_small = processor(text=task, images=dummy_small, return_tensors="pt")
        inputs_small = inputs_small.to(actual_device, dtype)

        with torch.no_grad():
            _ = model.generate(
                input_ids=inputs_small["input_ids"],
                pixel_values=inputs_small["pixel_values"],
                max_new_tokens=128,
                num_beams=1,
                do_sample=False,
            )

        # Прогрев 2: реальный размер SCADA-кадра (1280x768) —
        # именно такой размер кадра поступает в propose() после _resize_for_ocr.
        # Используем <OCR_WITH_REGION> — ту же таску, что и в propose()
        dummy_scada = Image.new("RGB", (1280, 768), color="black")
        task_region = "<OCR_WITH_REGION>"
        inputs_scada = processor(text=task_region, images=dummy_scada, return_tensors="pt")
        inputs_scada = inputs_scada.to(actual_device, dtype)

        with torch.no_grad():
            _ = model.generate(
                input_ids=inputs_scada["input_ids"],
                pixel_values=inputs_scada["pixel_values"],
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
            )

        # Прогрев 3: синхронизация CUDA для точного замера
        if actual_device.startswith("cuda"):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass  # Синхронизация не критична

        elapsed = time.perf_counter() - t0
        logger.info("Florence-2 прогрета за %.2fs на %s (SCADA warmup: 1280x768)", elapsed, actual_device)
        return elapsed

    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.warning("Ошибка при прогреве Florence-2 (%.2fs): %s", elapsed, e)
        return elapsed
