"""Асинхронный клиент для взаимодействия с VLM через llama-server.

Модуль предоставляет интерфейс для отправки multimodal запросов
к OpenAI-compatible API llama-server для анализа кадров видео.
"""

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


class VLMClient:
    """Асинхронный клиент для VLM inference через llama-server.

    Поддерживает отправку изображений в формате base64 JPEG
    и получение структурированных JSON-ответов от модели.

    Attributes:
        base_url: Базовый URL llama-server (например, http://localhost:8090).
        model_name: Имя модели для использования в API запросах.
        max_tokens: Максимальное количество токенов в ответе.
        temperature: Температура сэмплирования (0.0 - 1.0).
        client: Экземпляр httpx.AsyncClient для HTTP запросов.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        max_image_size: int = 640,
        top_p: float = 0.8,
        top_k: int = 20,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
    ) -> None:
        """Инициализирует клиент с заданными параметрами.

        Args:
            base_url: Базовый URL llama-server.
            model_name: Имя модели для API запросов.
            max_tokens: Максимальное количество токенов в ответе.
            temperature: Температура сэмплирования (Qwen3.5: 0.7 для general tasks).
            max_image_size: Максимальный размер стороны изображения (даунскейл).
            top_p: Nucleus sampling (Qwen3.5: 0.8).
            top_k: Top-k sampling (Qwen3.5: 20).
            presence_penalty: Штраф за повторение тем (Qwen3.5: 1.5).
            repetition_penalty: Множитель штрафа за повторение (Qwen3.5: 1.0).
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_image_size = max_image_size
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty

        # Таймаут 120 секунд для VLM inference (4B модель может быть медленной)
        timeout = httpx.Timeout(120.0, connect=10.0)
        self.client = httpx.AsyncClient(timeout=timeout)

        logger.info(
            "VLMClient инициализирован: url=%s, model=%s, max_tokens=%d, temp=%.2f, "
            "top_p=%.2f, top_k=%d, presence_penalty=%.1f, max_image_size=%d",
            self.base_url,
            self.model_name,
            self.max_tokens,
            self.temperature,
            self.top_p,
            self.top_k,
            self.presence_penalty,
            self.max_image_size,
        )
    def _encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """Кодирует numpy array в base64 JPEG строку с даунскейлом и сжатием.

        Выполняет даунскейл до max_image_size (640) и кодирует в JPEG с качеством 70
        для уменьшения размера payload и ускорения передачи.
        Оптимизировано для VLM: 640px + JPEG 70 даёт 3-5x ускорение по сравнению с 1024px + JPEG 75.

        Args:
            frame: Изображение в формате numpy array (BGR).

        Returns:
            Base64-encoded JPEG строка.

        Raises:
            ValueError: Если не удалось закодировать изображение.
        """
        # CLAHE ОТКЛЮЧЕН: искажает цвета SCADA (синий→серый, красный→оранжевый)
        # SCADA опирается на цветовую семантику: синий=актив, красный=авария, зеленый=норма
        #CLAHE: lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        #CLAHE: l, a, b = cv2.split(lab)
        #CLAHE: clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        #CLAHE: l = clahe.apply(l)
        #CLAHE: lab = cv2.merge([l, a, b])
        #CLAHE: frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Даунскейл если изображение слишком большое
        h, w = frame.shape[:2]
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            logger.info(
                "Resized frame from %dx%d to %dx%d",
                w, h, int(w * scale), int(h * scale),
            )

        # Кодируем в JPEG с качеством 100 (без сжатия для максимальной четкости SCADA)
        success, encoded_image = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100]
        )
        if not success:
            raise ValueError("Не удалось закодировать кадр в JPEG")

        base64_string = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
        return base64_string

    def _extract_json_from_response(self, content: str) -> dict[str, Any]:
        """Извлекает JSON из ответа модели с обработкой различных форматов.

        Поддерживает:
        - Прямой JSON: {"key": "value"}
        - Markdown code blocks: ```json\n{...}\n```
        - JSON внутри текста: текст {...} текст

        Args:
            content: Текстовый ответ от модели.

        Returns:
            Распарсенный JSON как dict.

        Raises:
            ValueError: Если не удалось извлечь и распарсить JSON.
        """
        return self._extract_json_from_response_static(content)

    @staticmethod
    def _extract_json_from_response_static(content: str) -> dict[str, Any]:
        """Статическая версия извлечения JSON — для использования из VLMClientDirect.

        Включает robust JSON repair для исправления типичных ошибок LLM:
        - Незакрытые скобки/кавычки
        - Trailing commas
        - Одинарные кавычки вместо двойных
        - Unescaped characters
        - Обрезанный JSON (max_tokens limit)

        Args:
            content: Текстовый ответ от модели.

        Returns:
            Распарсенный JSON как dict.

        Raises:
            ValueError: Если не удалось извлечь и распарсить JSON.
        """
        content = content.strip()

        # Попытка 1: Прямой парсинг
        try:
            result = json.loads(content)
            logger.debug("JSON extraction method: direct_parse, success=True")
            return result
        except json.JSONDecodeError:
            pass

        # Попытка 2: Извлечение из markdown code block ```json...```
        markdown_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        markdown_match = re.search(markdown_pattern, content, re.DOTALL)
        if markdown_match:
            json_str = markdown_match.group(1).strip()
            try:
                result = json.loads(json_str)
                logger.debug("JSON extraction method: markdown_code_block, success=True")
                return result
            except json.JSONDecodeError:
                # Markdown найден, но JSON битый — пробуем repair
                logger.debug("JSON in markdown block is malformed, attempting repair...")
                repaired = VLMClient._repair_json(json_str)
                if repaired:
                    try:
                        result = json.loads(repaired)
                        logger.debug("JSON extraction method: markdown+repair, success=True")
                        return result
                    except json.JSONDecodeError:
                        pass

        # Попытка 3: Поиск JSON объекта в тексте через regex
        # Ищем содержимое между первой { и последней }
        json_pattern = r"(\{.*\})"
        json_match = re.search(json_pattern, content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                result = json.loads(json_str)
                logger.debug("JSON extraction method: regex_brace_extraction, success=True")
                return result
            except json.JSONDecodeError:
                # JSON найден, но битый — пробуем repair
                logger.debug("Extracted JSON is malformed, attempting repair...")
                repaired = VLMClient._repair_json(json_str)
                if repaired:
                    try:
                        result = json.loads(repaired)
                        logger.debug("JSON extraction method: regex+repair, success=True")
                        return result
                    except json.JSONDecodeError:
                        pass

        # Попытка 4: Поиск массива JSON [...]
        array_pattern = r"(\[.*\])"
        array_match = re.search(array_pattern, content, re.DOTALL)
        if array_match:
            array_str = array_match.group(1).strip()
            try:
                parsed_array = json.loads(array_str)
                # Если это массив, обернуть в dict
                result = {"parameters": parsed_array}
                logger.debug("JSON extraction method: regex_array_extraction, success=True")
                return result
            except json.JSONDecodeError:
                pass

        # Попытка 5: Если JSON валидный но это массив (не объект), обернуть
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                # Модель вернула [{...}, {...}] вместо {"parameters": [{...}, {...}]}
                logger.warning(
                    "VLM returned array instead of object — wrapping in {'parameters': ...}. "
                    "Array length: %d, first_item_keys=%s",
                    len(parsed),
                    list(parsed[0].keys())[:5] if parsed and isinstance(parsed[0], dict) else "N/A",
                )
                result = {"parameters": parsed}
                return result
            elif isinstance(parsed, dict):
                return parsed
            else:
                logger.error(
                    "VLM returned unexpected JSON type: %s — content_preview=%s",
                    type(parsed).__name__,
                    content[:200],
                )
        except json.JSONDecodeError:
            pass

        logger.debug("JSON extraction method: all_methods_failed, content_preview=%s", content[:100])
        raise ValueError(f"Не удалось извлечь JSON из ответа: {content[:200]}...")

    @staticmethod
    def _repair_json(json_str: str) -> str | None:
        """Пытается исправить типичные ошибки в JSON от LLM.

        Исправляет:
        - Trailing commas: {"a": 1,}
        - Одинарные кавычки: {'a': 1}
        - Незакрытые строки: {"a": "hello
        - Незакрытые объекты/массивы: {"a": 1
        - Missing commas между объектами: } { → }, {
        - Unescaped newlines в строках
        - Комментарии // или /* */

        Args:
            json_str: Потенциально битый JSON.

        Returns:
            Исправленный JSON или None если не удалось исправить.
        """
        if not json_str:
            return None

        repaired = json_str

        # Шаг 1: Удалить комментарии (// и /* */)
        repaired = re.sub(r'//[^\n]*', '', repaired)  # Single-line comments
        repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)  # Multi-line comments

        # Шаг 2: Заменить одинарные кавычки на двойные (но не внутри строк!)
        # Простой подход: заменяем все ' на " если нет вложенных '
        if "'" in repaired and '"' not in repaired:
            repaired = repaired.replace("'", '"')

        # Шаг 3: Исправить missing commas между объектами в массивах
        # Паттерн: } { или } \n { без запятой
        repaired = re.sub(r'\}\s*\{', '}, {', repaired)  # } { → }, {
        repaired = re.sub(r'\]\s*\[', '], [', repaired)  # ] [ → ], [
        
        # Шаг 3.5: Исправить missing commas после значений перед ключами
        # Паттерн: "value": 123\n"label" → "value": 123,\n"label"
        # Это происходит когда LLM забывает запятую между полями объекта
        repaired = re.sub(r'(:\s*(?:\d+|true|false|null|".*?"))\s*\n\s*"', r'\1,\n"', repaired)

        # Шаг 4: Удалить trailing commas перед } или ]
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

        # Шаг 5: Экранировать незакрытые строки
        # Найти строки без закрывающей кавычки
        lines = repaired.split('\n')
        fixed_lines = []
        in_string = False
        for line in lines:
            # Простой подсчет кавычек (не идеален, но работает для большинства случаев)
            quote_count = line.count('"') - line.count('\\"')
            if quote_count % 2 == 1:  # Нечетное количество = незакрытая строка
                if in_string:
                    # Закрыть строку в конце линии
                    line = line.rstrip() + '"'
                    in_string = False
                else:
                    in_string = True
            fixed_lines.append(line)
        repaired = '\n'.join(fixed_lines)

        # Шаг 6: Добавить недостающие закрывающие скобки
        # Подсчитать баланс скобок
        brace_count = 0
        bracket_count = 0

        for char in repaired:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1

        # Добавить недостающие закрывающие скобки
        closing = ''
        if bracket_count > 0:
            closing = ']' * bracket_count + closing
        if brace_count > 0:
            closing = '}' * brace_count + closing

        if closing:
            repaired = repaired.rstrip() + closing
            logger.debug("JSON repair: added closing brackets/braces: %s", closing)

        # Шаг 7: Попытка исправить missing commas перед закрывающими скобками
        # Иногда LLM забывает запятую: "value": 123\n} → "value": 123,\n}
        # Это сложно сделать правильно, но можно попробовать эвристику
        # Пропускаем чтобы не сломать валидный JSON

        # Финальная проверка
        try:
            json.loads(repaired)
            logger.info("JSON repair SUCCESS — fixed common LLM errors")
            return repaired
        except json.JSONDecodeError as e:
            logger.debug("JSON repair FAILED — %s", str(e)[:100])
            
            # Попробовать более агрессивный repair: обрезать до последней валидной позиции
            logger.debug("Attempting aggressive repair: truncate to last valid JSON...")
            truncated = VLMClient._truncate_to_valid_json(repaired)
            if truncated:
                try:
                    parsed = json.loads(truncated)
                    # Убедиться что это dict, не list
                    if isinstance(parsed, list):
                        logger.warning(
                            "Truncated JSON is array — wrapping in {'parameters': ...}"
                        )
                        parsed = {"parameters": parsed}
                    logger.info("JSON repair SUCCESS (aggressive truncation)")
                    return json.dumps(parsed)  # Вернуть как строку
                except json.JSONDecodeError:
                    pass
            
            return None

    @staticmethod
    def _truncate_to_valid_json(json_str: str) -> str | None:
        """Пытается найти валидный JSON путем постепенного усечения.

        Стратегия:
        - Найти последний полный объект в массиве parameters
        - Обрезать всё после него
        - Закрыть скобки корректно

        Args:
            json_str: Потенциально битый JSON.

        Returns:
            Валидный JSON или None.
        """
        # Найти позицию последнего полностью закрытого параметра
        # Паттерн: {"label": ..., "value": ..., "in_range": true}
        last_complete_param = None
        
        # Ищем завершенные объекты параметров
        param_pattern = r'\{[^{}]*"label"[^{}]*"value"[^{}]*"in_range"[^{}]*\}'
        matches = list(re.finditer(param_pattern, json_str, re.DOTALL))
        
        if matches:
            # Взять последний полный параметр
            last_match = matches[-1]
            last_complete_param = last_match.end()
            
            # Найти позицию этого параметра в массиве
            # Обрезать JSON после этого параметра
            truncated = json_str[:last_complete_param]
            
            # Теперь нужно закрыть структуру корректно
            # Найти начало массива parameters
            params_array_start = truncated.rfind('"parameters"')
            if params_array_start == -1:
                return None
            
            # Найти открывающую скобку массива [
            array_open = truncated.find('[', params_array_start)
            if array_open == -1:
                return None
            
            # Построить валидный JSON
            result = json_str[:array_open+1]  # Включая [
            # Добавить все полные параметры до последнего
            result += truncated[array_open+1:last_complete_param]
            result += ']}'  # Закрыть массив и объект
            
            return result
        
        return None

    async def _make_request_with_retry(
        self,
        endpoint: str,
        payload: dict[str, Any],
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> dict[str, Any]:
        """Выполняет HTTP POST запрос с экспоненциальным backoff retry.

        Args:
            endpoint: API endpoint (относительный путь).
            payload: JSON payload для отправки.
            max_retries: Максимальное количество попыток.
            base_delay: Начальная задержка между попытками в секундах.

        Returns:
            JSON ответ от сервера.

        Raises:
            httpx.HTTPError: Если все попытки исчерпаны.
        """
        import time as time_module

        url = f"{self.base_url}{endpoint}"
        last_exception: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                last_exception = e
                # WARNING: Retry event with HTTP status error
                logger.warning(
                    "Retry event: attempt=%d/%d, error_type=http_status_error, status_code=%d, wait_time=%.1fs",
                    attempt + 1,
                    max_retries,
                    e.response.status_code,
                    base_delay * (2**attempt) if attempt < max_retries - 1 else 0,
                )
                logger.warning(
                    "HTTP ошибка (попытка %d/%d): %s - %s",
                    attempt + 1,
                    max_retries,
                    e.response.status_code,
                    e.response.text[:200],
                )
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.info("Повтор через %.1f секунд...", delay)
                    await asyncio.sleep(delay)

            except httpx.RequestError as e:
                last_exception = e
                # WARNING: Retry event with connection error
                logger.warning(
                    "Retry event: attempt=%d/%d, error_type=request_error, error_msg=%s, wait_time=%.1fs",
                    attempt + 1,
                    max_retries,
                    str(e)[:100],
                    base_delay * (2**attempt) if attempt < max_retries - 1 else 0,
                )
                logger.warning(
                    "Ошибка соединения (попытка %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    str(e),
                )
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.info("Повтор через %.1f секунд...", delay)
                    await asyncio.sleep(delay)

        raise last_exception or httpx.RequestError("Все попытки исчерпаны")

    async def analyze_frame(
        self,
        frame: np.ndarray,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Анализирует один кадр с помощью VLM.

        Args:
            frame: Изображение в формате numpy array (BGR).
            system_prompt: Системный промпт для модели.
            user_prompt: Пользовательский промпт для модели.

        Returns:
            Распарсенный JSON ответ от модели.

        Raises:
            ValueError: Если не удалось обработать кадр или ответ.
            httpx.HTTPError: Если запрос не удался после всех попыток.
        """
        # КРИТИЧНО: DEBUG — сохраняем кадр для визуальной проверки
        import os
        debug_dir = "debug_frames"
        os.makedirs(debug_dir, exist_ok=True)
        
        # DEBUG: Request preparation
        url = f"{self.base_url}/v1/chat/completions"
        logger.debug(
            "Request preparation: url=%s, model=%s, max_tokens=%d, temperature=%.2f",
            url,
            self.model_name,
            self.max_tokens,
            self.temperature,
        )

        # Кодируем кадр в base64
        try:
            # DEBUG: Image encoding before
            frame_h, frame_w = frame.shape[:2]
            logger.debug(
                "Image encoding: dimensions=%dx%d",
                frame_w,
                frame_h,
            )

            base64_image = self._encode_frame_to_base64(frame)

            # DEBUG: Image encoding result
            jpeg_size_kb = len(base64_image) * 3 // 4 / 1024  # base64 overhead
            logger.debug(
                "Image encoding: jpeg_size=%.1fKB, base64_size=%.1fKB",
                jpeg_size_kb,
                len(base64_image) / 1024,
            )
            
            # КРИТИЧНО: Сохраняем кадр для визуальной проверки (каждый 10-й)
            if not hasattr(self, '_debug_frame_counter'):
                self._debug_frame_counter = 0
            self._debug_frame_counter += 1
            
            # Сохраняем каждый кадр (для первых 5 кадров)
            if self._debug_frame_counter <= 5:
                debug_path = os.path.join(debug_dir, f"frame_{self._debug_frame_counter:03d}_{frame_w}x{frame_h}.jpg")
                cv2.imwrite(debug_path, frame)
                logger.warning(
                    "DEBUG FRAME SAVED: %s (original %dx%d, JPEG %.1fKB) — CHECK THIS FILE!",
                    debug_path, frame_w, frame_h, jpeg_size_kb,
                )
        except ValueError as e:
            logger.error("Ошибка кодирования кадра: %s", str(e))
            raise

        # Формируем OpenAI-compatible сообщение
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "stream": False,
        }

        # КРИТИЧНО: Логируем первые 200 символов промпта для проверки
        logger.warning(
            "VLM PROMPT PREVIEW (first 200 chars):\n%s",
            user_prompt[:200],
        )
        
        logger.debug(
            "Отправка запроса к VLM: model=%s, prompt_len=%d",
            self.model_name,
            len(user_prompt),
        )

        # DEBUG: Request sent timestamp
        request_start = time.perf_counter()
        logger.debug("Request sent: timestamp=%.6f", request_start)

        # Выполняем запрос с retry
        response = await self._make_request_with_retry("/v1/chat/completions", payload)

        # INFO: Response received with timing
        response_elapsed_ms = (time.perf_counter() - request_start) * 1000
        logger.info(
            "Response received: http_status=200, response_time=%.1fms",
            response_elapsed_ms,
        )

        # Извлекаем контент из ответа
        try:
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("Ответ не содержит choices")

            message = choices[0].get("message", {})
            content = message.get("content", "")

            if not content:
                raise ValueError("Ответ не содержит content")

        except (KeyError, IndexError) as e:
            raise ValueError(f"Некорректная структура ответа: {str(e)}") from e

        # DEBUG: Token usage from response
        usage = response.get("usage", {})
        if usage:
            logger.debug(
                "Token usage: prompt_tokens=%d, completion_tokens=%d, total_tokens=%d",
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("total_tokens", 0),
            )

        # Парсим JSON из контента
        result = self._extract_json_from_response(content)
        
        # КРИТИЧНО: Логируем первый параметр для проверки reality check
        if isinstance(result, dict) and "parameters" in result:
            params = result.get("parameters", [])
            if params:
                first_param = params[0]
                logger.warning(
                    "VLM REALITY CHECK: First param: label='%s', value='%s', unit='%s'",
                    first_param.get("label", "N/A"),
                    first_param.get("value", "N/A"),
                    first_param.get("unit", "N/A"),
                )
                logger.warning(
                    "VLM RESPONSE SUMMARY: total_params=%d, first_value_type=%s",
                    len(params),
                    type(first_param.get("value")).__name__,
                )
        
        logger.debug("VLM ответ успешно распарсен")

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
        url = f"{self.base_url}/v1/chat/completions"

        # Кодируем все изображения в base64
        image_contents: list[dict[str, Any]] = []
        for i, frame in enumerate(frames):
            base64_image = self._encode_frame_to_base64(frame)
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
            })

        # Добавляем текстовый промпт после всех изображений
        image_contents.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_contents},
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "stream": False,
        }

        logger.info(
            "Multi-image request: %d images, model=%s",
            len(frames), self.model_name,
        )

        request_start = time.perf_counter()
        response = await self._make_request_with_retry("/v1/chat/completions", payload)
        response_elapsed_ms = (time.perf_counter() - request_start) * 1000

        logger.info(
            "Multi-image response: %.1fms, images=%d",
            response_elapsed_ms, len(frames),
        )

        # Извлекаем контент
        try:
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("Ответ не содержит choices")
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if not content:
                raise ValueError("Ответ не содержит content")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Некорректная структура ответа: {str(e)}") from e

        result = self._extract_json_from_response(content)

        # Помечаем что это residual (zone_id=0)
        if isinstance(result, dict) and "zone_id" not in result:
            result["zone_id"] = 0

        return result

    async def analyze_frame_batch(
        self,
        frames: list[tuple[np.ndarray, str]],
        system_prompt: str,
        user_prompt: str,
    ) -> list[dict[str, Any]]:
        """Анализирует батч кадров последовательно (batch size 1 для VRAM).

        Args:
            frames: Список кортежей (frame, timestamp).
            system_prompt: Системный промпт для модели.
            user_prompt: Пользовательский промпт для модели.

        Returns:
            Список результатов с добавленным полем timestamp.

        Note:
            Обработка последовательная для экономии VRAM (8GB constraint).
        """
        results: list[dict[str, Any]] = []

        for i, (frame, timestamp) in enumerate(frames):
            logger.info(
                "Обработка кадра %d/%d (timestamp: %s)",
                i + 1,
                len(frames),
                timestamp,
            )

            try:
                result = await self.analyze_frame(frame, system_prompt, user_prompt)
                result["timestamp"] = timestamp
                results.append(result)

            except Exception as e:
                logger.error(
                    "Ошибка обработки кадра %d (timestamp: %s): %s",
                    i + 1,
                    timestamp,
                    str(e),
                )
                # Добавляем результат с ошибкой для сохранения порядка
                results.append({
                    "timestamp": timestamp,
                    "error": str(e),
                    "parameters": [],
                })

        return results

    async def health_check(self) -> bool:
        """Проверяет доступность llama-server.

        Returns:
            True если сервер доступен (HTTP 200), иначе False.
        """
        health_url = f"{self.base_url}/health"
        # DEBUG: Health check URL and attempt
        logger.debug("Health check: url=%s", health_url)

        try:
            response = await self.client.get(
                health_url,
                timeout=5.0,
            )
            result = response.status_code == 200
            # DEBUG: Health check result
            logger.debug(
                "Health check: url=%s, status_code=%d, result=%s",
                health_url,
                response.status_code,
                "healthy" if result else "unhealthy",
            )
            return result

        except httpx.RequestError as e:
            # DEBUG: Health check failure
            logger.debug("Health check: url=%s, error=%s, result=unhealthy", health_url, str(e))
            logger.debug("Health check failed: %s", str(e))
            return False

    async def close(self) -> None:
        """Закрывает HTTP клиент и освобождает ресурсы."""
        await self.client.aclose()
        logger.info("VLMClient закрыт")

    async def __aenter__(self) -> "VLMClient":
        """Асинхронный контекстный менеджер - вход."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Асинхронный контекстный менеджер - выход."""
        await self.close()


def get_vlm_client() -> VLMClient:
    """Создает VLM клиент из настроек приложения.

    Пытается сначала использовать VLMClientDirect (native llama-cpp-python),
    если llama-cpp-python установлен. Иначе fallback на HTTP VLMClient.

    Returns:
        Настроенный экземпляр VLMClient или VLMClientDirect.

    Example:
        >>> client = get_vlm_client()
        >>> result = await client.analyze_frame(frame, system_prompt, user_prompt)
    """
    from app.config import settings

    # Пытаемся использовать VLMClientDirect (native, быстрее)
    use_direct = getattr(settings, "vlm_use_direct", True)
    if use_direct:
        try:
            from app.core.vlm_client_direct import VLMClientDirect
            logger.info("Используется VLMClientDirect (native llama-cpp-python)")
            return VLMClientDirect()  # type: ignore[return-value]
        except (ImportError, FileNotFoundError) as e:
            logger.warning(
                "VLMClientDirect недоступен (%s), fallback на HTTP VLMClient", e
            )

    # Fallback: HTTP-клиент через llama-server
    base_url = getattr(settings, "vlm_base_url", "http://localhost:8090")
    model_name = getattr(settings, "vlm_model_name", "Qwen3.5-4B")
    max_tokens = getattr(settings, "vlm_max_tokens", 1024)
    temperature = getattr(settings, "vlm_temperature", 0.0)
    max_image_size = getattr(settings, "vlm_max_image_size", 640)

    logger.info("Используется HTTP VLMClient (url=%s)", base_url)
    return VLMClient(
        base_url=base_url,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        max_image_size=max_image_size,
    )
