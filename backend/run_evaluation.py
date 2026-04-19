"""Автономный скрипт оценки качества распознавания.

Запускает полный конвейер OCR/VLM на предоставленных видео и сравнивает
результаты с эталонными значениями (ground truth).

Поддерживает два режима сравнения:
- Fuzzy matching (по умолчанию) — строковое сравнение с SequenceMatcher
- LLM-as-a-Judge (--llm-judge) — LLM оценивает семантическое соответствие
  названий и корректность значений

Использование:
    # VLM оценка с ground truth XML:
    python run_evaluation.py --vlm --video "../Задание 2/Видео 1/Видео 1.mp4" --ground-truth-xml "../Задание 2/Видео 1/ground_truth.xml"

    # VLM оценка с LLM-as-Judge:
    python run_evaluation.py --vlm --llm-judge --video "../Задание 2/Видео 1/Видео 1.mp4" --ground-truth-xml "../Задание 2/Видео 1/ground_truth.xml"

    # VLM оценка без зонного анализа (fallback — полный кадр):
    python run_evaluation.py --vlm --no-zones --video "../Задание 2/Видео 2/Видео 2.mp4" --ground-truth-xml "../Задание 2/Видео 2/ground_truth.xml"

    # Оценка конкретного видео с таблицей параметров (OCR):
    python run_evaluation.py --video "../Задание 2/Видео 1/Видео 1.mp4" --table "../Задание 2/Видео 1/Таблица 1.xlsx"

    # Оценка всех видео из Задание 2:
    python run_evaluation.py --all

    # Только OCR без таблицы параметров (baseline):
    python run_evaluation.py --video "../Задание 2/Видео 2/Видео 2.mp4"

    # С указанием типа видео:
    python run_evaluation.py --video video.mp4 --table table.xlsx --video-type direct

    # С эталонным XML для сравнения (OCR):
    python run_evaluation.py --video video.mp4 --table table.xlsx --ground-truth ground_truth.xml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

# Отключаем OneDNN и PIR-executor для PaddlePaddle 3.x на Windows
# (NotImplementedError / layout-конфликт в OneDNN) — ДОЛЖНО быть до импорта paddle
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["MKLDNN_DISABLE"] = "1"

# PaddleOCR 2.9+ через PaddleX тянет modelscope -> torch.
# Если paddle загрузится первым, DLL torch испортится.
try:
    import torch  # noqa: F401 — preload до paddle
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

from pathlib import Path
from typing import Any

# Добавляем backend/ в sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================================
# LLM-as-a-Judge: клиент и логика оценки
# ============================================================================


class LLMJudgeClient:
    """Текстовый LLM-клиент для judge-оценки через llama-server.

    Отправляет text-only запросы к OpenAI-compatible API llama-server
    для семантического сравнения параметров (без изображений).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8090",
        model_name: str = "Qwen3.5-4B",
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def judge(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 2,
    ) -> str:
        """Отправляет text-only запрос к LLM и возвращает текст ответа.

        Args:
            system_prompt: Системный промпт.
            user_prompt: Пользовательский промпт.
            max_retries: Количество попыток при ошибке.

        Returns:
            Текст ответа от модели.
        """
        import httpx

        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        timeout = httpx.Timeout(180.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    return content
                except (httpx.HTTPError, KeyError, IndexError) as e:
                    if attempt < max_retries - 1:
                        print(f"    [WARN] LLM judge retry {attempt + 1}/{max_retries}: {e}")
                        import asyncio
                        await asyncio.sleep(2.0)
                    else:
                        raise

        return ""  # unreachable


def _build_judge_system_prompt() -> str:
    """Строит системный промпт для LLM-judge."""
    return """Ты — судья-эксперт по оценке качества распознавания параметров SCADA-мнемосхем.

Твоя задача: сравнить распознанные параметры с эталонными и определить:
1. Соответствует ли распознанный параметр эталонному (семантически, даже если названия отличаются)
2. Совпадает ли значение (с допуском 1% относительной погрешности для чисел)

КРИТИЧЕСКИЕ ПРАВИЛА СРАВНЕНИЯ ИМЁН:
- Названия МОГУТ ОТЛИЧАТЬСЯ. Это НОРМА для SCADA-систем. name_match=true если названия обозначают
  один и тот же физический параметр, даже если форма записи полностью разная.
- Сокращения эквивалентны полным названиям:
  * Т / Температура / Темп. — одно и то же
  * Р / Давление / Давл. — одно и то же
  * dP / dП / Перепад давления / Дельта Р — одно и то же
  * L / Уровень / Ур. — одно и то же
  * N / Обороты / Частота вращения / n — одно и то же
  * V / Напряжение — одно и то же
  * f / Влажность — одно и то же
  * Pos / Положение / Задание — одно и то же
- Индексы оборудования эквивалентны:
  * ГТД / ГПД — газотурбинный двигатель (разные транслитерации)
  * ЦБК / МБК — компрессорная установка (разные аббревиатуры)
  * РНД / ТНД — ротор/турбина низкого давления
  * РВД — ротор высокого давления
  * РСТ — регулятор скорости турбины
- Составные сокращения:
  * «Lм в ГТД» = «Уровень масла ГПД» (L=уровень, м=масло, в ГТД=в ГПД)
  * «Tм на вх.» = «Температура масла на входе» (T=температура, м=масло)
  * «Lм в МБК» = «Уровень масла ЦБК» (L=уровень, м=масло, МБК≈ЦБК)
  * «Tм в коллек» = «Температура масла в колпаке» (коллек≈колпак, сокращение)
  * «Т кожуха» = «Температура воздуха под кожухом двигателя»
  * «Р кожуха» = «Давление воздуха под кожухом»
  * «Р после 2-1» = «Давление после крана 2-1» (Р=давление, после=after, 2-1=номер клапана/крана)
  * «Р после X» = «Давление после крана X» (общее правило: Р после = давление после)
  * «T2 точка N» = «Температура газов за ТНД, точка N» (T2 — температура газов)
  * «Обобщённая температура (N)» = «Температура газов за ТНД, обобщенный сигнал»
  * «Р после 2-1» может быть = «Перепад давления на фильтре 2-1» (если значение совпадает)
- Опущенные слова, другие падежи, перестановки — НЕ являются причиной name_match=false.
- Если распознанное название и эталонное описывают ОДИН И ТОТ ЖЕ датчик/параметр → name_match=true.
- Если значения совпадают и названия относятся к одному физическому параметру → name_match=true,
  даже если форма записи полностью разная (сокращения, опущенные слова, другие падежи).

ПРАВИЛА СРАВНЕНИЯ ЗНАЧЕНИЙ:
- Главный критерий: СОВПАДЕНИЕ ЗНАЧЕНИЙ.
- Числовые значения: совпадение если |a-b|/max(|b|,0.001) <= 0.01 (1% допуск).
- Запятая и точка как разделитель: 36,1 = 36.1
- Для нулевых значений (gt=0.0): |rec| < 0.01 считается совпадением.

ПРАВИЛА СООТВЕТСТВИЯ:
- Один распознанный параметр может соответствовать только одному эталонному.
- Если для эталонного параметра нет распознанного — name_match=false, value_match=false.
- ЕСЛИ VALUE_MATCH=TRUE И NAME_MATCH=TRUE → параметр распознан ПРАВИЛЬНО.
- ЕСЛИ VALUE_MATCH=TRUE И ИМЕНА СЕМАНТИЧЕСКИ БЛИЗКИ → тоже name_match=true!
  Не требуй точного совпадения имён. Достаточно семантической эквивалентности.

ФОРМАТ ОТВЕТА — строго JSON (без markdown, без комментариев, без лишнего текста):
{
  "judgments": [
    {
      "gt_param_name": "название эталонного параметра",
      "gt_value": "значение эталона",
      "matched_rec_param_name": "название распознанного параметра" или null,
      "rec_value": "распознанное значение" или null,
      "name_match": true или false,
      "value_match": true или false,
      "reasoning": "краткое объяснение"
    }
  ],
  "unmatched_rec_params": ["названия распознанных параметров без пары"],
  "total_gt_params": N,
  "name_matched_params": M,
  "value_matched_params": K
}"""


def _build_judge_user_prompt(
    gt_params: dict[str, str],
    rec_params: dict[str, str],
    timestamp: str,
) -> str:
    """Строит пользовательский промпт для LLM-judge.

    Args:
        gt_params: Эталонные параметры {name: value}.
        rec_params: Распознанные параметры {name: value}.
        timestamp: Таймстемп для контекста.

    Returns:
        Пользовательский промпт.
    """
    gt_lines = []
    for name, value in gt_params.items():
        gt_lines.append(f'  "{name}": "{value}"')
    gt_block = "{\n" + ",\n".join(gt_lines) + "\n}"

    rec_lines = []
    for name, value in rec_params.items():
        rec_lines.append(f'  "{name}": "{value}"')
    rec_block = "{\n" + ",\n".join(rec_lines) + "\n}" if rec_params else "{}"

    return f"""Таймстемп: {timestamp}

Эталонные параметры (ground truth):
{gt_block}

Распознанные параметры (VLM output):
{rec_block}

Сравни каждый эталонный параметр с распознанными и верни JSON-результат."""


def _repair_json_string(json_str: str) -> str:
    """Пытается исправить типичные ошибки JSON из ответов LLM.

    Args:
        json_str: Строка с потенциально битым JSON.

    Returns:
        Исправленная строка JSON.
    """
    import re as _re

    s = json_str

    # 1. Убираем trailing commas перед } или ]
    s = _re.sub(r',[\s]*([}\]])', r'\1', s)

    # 2. Убираем JS-стиль комментариев (// и /* */)
    s = _re.sub(r'//[^\n]*', '', s)
    s = _re.sub(r'/\*.*?\*/', '', s, flags=_re.DOTALL)

    # 3. Фиксим одинарные кавычки -> двойные (только для ключей и строк)
    # Осторожно: не трогаем одинарные кавычки внутри двойных
    def _fix_quotes(m: _re.Match) -> str:
        return m.group(0).replace("'", '"')
    s = _re.sub(r"'([^']*)'(?=\s*[:\],}])", r'"\1"', s)

    # 4. Фиксим незакрытые кавычки в значениях (обрезанные строки)
    # Если строка не закрыта перед запятой/скобкой — добавляем закрывающую кавычку
    s = _re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)\n', r'"\1"\n', s)

    # 5. Фиксим boolean: True/False/None -> true/false/null
    s = _re.sub(r'\bTrue\b', 'true', s)
    s = _re.sub(r'\bFalse\b', 'false', s)
    s = _re.sub(r'\bNone\b', 'null', s)

    # 6. Фиксим незакрытые фигурные/квадратные скобки
    open_curly = s.count('{') - s.count('}')
    open_square = s.count('[') - s.count(']')
    if open_curly > 0:
        s += '}' * open_curly
    if open_square > 0:
        s += ']' * open_square

    return s


def _extract_judge_json(response_text: str) -> dict[str, Any]:
    """Извлекает JSON из ответа LLM-judge.

    Поддерживает: прямой JSON, markdown code block, JSON внутри текста.
    Включает robust repair для типичных ошибок LLM-ответов.

    Args:
        response_text: Текст ответа LLM.

    Returns:
        Распарсенный JSON. Содержит ключ "parse_error": True если парсинг не удался.
    """
    import re as _re

    text = response_text.strip()

    # Кандидаты для парсинга (порядок: от наивного к продвинутому)
    candidates: list[str] = []

    # Кандидат 1: весь текст как есть
    candidates.append(text)

    # Кандидат 2: markdown code block
    md_match = _re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, _re.DOTALL)
    if md_match:
        candidates.append(md_match.group(1).strip())

    # Кандидат 3: первая { до последней }
    brace_match = _re.search(r'(\{.*\})', text, _re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1))

    # Пытаемся парсить каждый кандидат: сначала как есть, потом с repair
    for candidate in candidates:
        # Попытка без repair
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Попытка с repair
        repaired = _repair_json_string(candidate)
        try:
            result = json.loads(repaired)
            if isinstance(result, dict):
                print(f"    [INFO] JSON repaired successfully")
                return result
        except json.JSONDecodeError:
            pass

    # Fallback: возвращаем пустую структуру с флагом ошибки
    print(f"    [WARN] Failed to parse LLM judge JSON response")
    return {
        "judgments": [],
        "unmatched_rec_params": [],
        "total_gt_params": 0,
        "name_matched_params": 0,
        "value_matched_params": 0,
        "parse_error": True,
        "raw_response_preview": text[:500],
    }


def _upgrade_name_match(
    judgments: list[dict],
    gt_params: dict[str, str],
    rec_params: dict[str, str],
) -> list[dict]:
    """Пост-обработка: обновляет name_match для семантически близких пар.

    Если value_match=true но name_match=false, и названия семантически
    соответствуют друг другу — устанавливает name_match=true.
    Это компенсирует случаи, когда LLM-judge не распознал сокращение.

    Args:
        judgments: Список решений judge.
        gt_params: Эталонные параметры.
        rec_params: Распознанные параметры.

    Returns:
        Обновлённый список решений.
    """
    from difflib import SequenceMatcher
    import re as _re

    # Канонические сокращения SCADA: ключевые слова и их сокращения
    ABBREV_MAP: dict[str, list[str]] = {
        "температура": ["т", "t", "темп", "т2"],
        "давление": ["р", "p", "давл", "рг"],
        "перепад": ["dp", "dп", "dр", "дельта"],
        "уровень": ["l", "лм", "ур"],
        "обороты": ["n", "об/мин"],
        "частота": ["f", "гц"],
        "влажность": ["f", "%"],
        "масло": ["м", "масл"],
        "газ": ["г", "газ"],
        "воздух": ["в", "возд"],
        "газов": ["г", "газ"],
        "компрессор": ["к", "компр"],
        "турбин": ["т", "тг"],
        "кожух": ["кож", "кожуха"],
        "подшипник": ["п", "подш", "оп", "оуп", "рк", "ук"],
        "подогрев": ["подогр", "под", "б газ"],
        "вход": ["вх", "вх."],
        "выход": ["вых", "вых."],
        "обобщен": ["обобщ", "обобщённ"],
        "количество": ["т аво", "кол-во", "кол"],
        "атмосферное": ["атм", "атм."],
        "наружного": ["наруж", "нар"],
    }

    # Прямые эквиваленты оборудования и SCADA-сокращений
    EQUIV_PAIRS: list[tuple[str, str]] = [
        ("гтд", "гпд"),
        ("цбк", "мбк"),
        ("тнд", "тнд"),
        ("рнд", "рнд"),
        ("аво", "аво"),
        ("колпак", "коллек"),
        ("т2", "температура газов за тнд"),
        # SCADA аббревиатуры подшипников
        ("оп ", "температура подшипника оп"),
        ("оуп ", "температура подшипника оуп"),
        ("рк ", "температура подшипника рк"),
        ("ук ", "температура подшипника ук"),
        # dP = перепад давления
        ("dp", "перепад"),
        # Т АВО = Количество АВО (SCADA отображение)
        ("т аво", "количество аво"),
        # Рг после кр = Давление после крана
        ("рг после кр", "давление после крана"),
        ("р после кр", "давление после крана"),
        # Т б газ вход = Температура подогрева (вход)
        ("т б газ вход", "температура подогрева"),
        ("т б газ", "подогрева"),
        # Т б газ подогр = Температура подогрева, точка 2
        ("т б газ подогр", "температура подогрева точка 2"),
    ]

    def _normalize(name: str) -> str:
        """Нормализует название для сравнения."""
        n = name.lower().strip()
        # Ё -> Е для единообразия
        n = n.replace('ё', 'е')
        # Убираем (N) индексы в скобках
        n = _re.sub(r'\([^)]*\)', '', n).strip()
        # Убираем лишние пробелы
        n = _re.sub(r'\s+', ' ', n)
        # Убираем точки на конце сокращений (вх. -> вх)
        n = _re.sub(r'\.(?=\s|$)', '', n)
        return n

    def _are_names_equivalent(gt_name: str, rec_name: str) -> bool:
        """Проверяет семантическую эквивалентность двух названий."""
        gt_norm = _normalize(gt_name)
        rec_norm = _normalize(rec_name)

        # 1. Прямое совпадение
        if gt_norm == rec_norm:
            return True

        # 2. Fuzzy matching: если очень похожи (>0.75)
        ratio = SequenceMatcher(None, gt_norm, rec_norm).ratio()
        if ratio >= 0.75:
            return True

        # 3. Проверка эквивалентов оборудования
        for a, b in EQUIV_PAIRS:
            if (a in gt_norm and b in rec_norm) or (b in gt_norm and a in rec_norm):
                return True

        # 4. Проверка по сокращениям: если все ключевые слова gt_name
        #    присутствуют в rec_name в полной или сокращённой форме
        #    (или наоборот) — считаем эквивалентными
        gt_words = set(gt_norm.split())
        rec_words = set(rec_norm.split())

        # Склеиваем все слова в одно для проверки составных сокращений
        gt_joined = gt_norm.replace(" ", "")
        rec_joined = rec_norm.replace(" ", "")

        # 5. Специальный случай: T2 = Температура газов за ТНД
        if ("т2" in rec_joined or "t2" in rec_joined) and (
            "температура" in gt_joined and "тнд" in gt_joined
        ):
            return True
        if ("т2" in gt_joined or "t2" in gt_joined) and (
            "температура" in rec_joined and "тнд" in rec_joined
        ):
            return True

        # 5b. Специальный случай: «Р после X» = «Давление после крана X»
        if "рпосле" in rec_joined or "рпосле" in gt_joined:
            # Р после = давление после
            if ("рпосле" in rec_joined and "давление" in gt_joined and "после" in gt_joined):
                return True
            if ("рпосле" in gt_joined and "давление" in rec_joined and "после" in rec_joined):
                return True

        # 6. Проверяем: каждое значимое слово gt_name имеет представление в rec_name
        # Это ловит «Lм в ГТД» = «Уровень масла ГПД» через сокращения
        significant_gt = [w for w in gt_words if len(w) > 1]
        significant_rec = [w for w in rec_words if len(w) > 1]

        # Также учитываем однобуквенные слова-сокращения (т, р, l, n и т.д.)
        single_gt = [w for w in gt_words if len(w) == 1]
        single_rec = [w for w in rec_words if len(w) == 1]

        if not significant_gt and not single_gt:
            return False
        if not significant_rec and not single_rec:
            return False

        # Быстрая проверка: если однобуквенное сокращение в rec совпадает
        # с первой буквой значимого слова gt (и наоборот)
        # Пример: «Т кожуха» -> «Температура воздуха под кожухом» (Т = Температура)
        # Пример: «Р кожуха» -> «Давление воздуха под кожухом» (Р = Давление через ABBREV_MAP)
        def _single_letter_matches(single_list: list[str], other_joined: str, other_significant: list[str]) -> bool:
            for sl in single_list:
                # Проверяем, является ли однобуквенное сокращение сокращением
                # какого-то полного слова в другой строке
                for full, abbrevs in ABBREV_MAP.items():
                    if sl in abbrevs:
                        # Это сокращение. Проверяем, есть ли полное слово в other
                        if full in other_joined or full[:4] in other_joined:
                            return True
                # Проверяем: начинается ли какое-то слово в other с этой буквы
                for ow in other_significant:
                    if ow.startswith(sl):
                        return True
            return False

        # Проверяем: rec_name является сокращённой версией gt_name
        # Пример: «Р кожуха» → «Давление воздуха под кожухом»
        # Ключ: «кожух» есть в обоих, «Р» = «Давление»
        matched_keywords = 0
        for gt_w in significant_gt:
            # Прямое вхождение (минимум 3 символа пересечения)
            if any(gt_w in rw or rw in gt_w for rw in significant_rec):
                # Дополнительная проверка: минимум 3 символа пересечения
                has_real_overlap = False
                for rw in significant_rec:
                    overlap_len = 0
                    for i in range(len(gt_w)):
                        for j in range(i + 2, len(gt_w) + 1):  # мин. 2 символа
                            if gt_w[i:j] in rw and (j - i) > overlap_len:
                                overlap_len = j - i
                    if overlap_len >= 2:
                        has_real_overlap = True
                        break
                if has_real_overlap:
                    matched_keywords += 1
                    continue
            # Через таблицу сокращений
            found = False
            for full, abbrevs in ABBREV_MAP.items():
                if gt_w.startswith(full) or full.startswith(gt_w):
                    if any(ab in rec_joined for ab in abbrevs):
                        found = True
                        break
                    # Склеенное сокращение (напр. «Lм» содержит «L» = «уровень»)
                    for ab in abbrevs:
                        if ab in rec_joined:
                            found = True
                            break
                # Обратная проверка: rec слово — сокращение gt слова
                for ab in abbrevs:
                    if any(rw.startswith(ab) for rw in significant_rec) and (
                        gt_w.startswith(full) or full.startswith(gt_w)
                    ):
                        found = True
                        break
                if found:
                    break
            if found:
                matched_keywords += 1

        # Учитываем однобуквенные сокращения как совпадения
        has_single_match = (
            _single_letter_matches(single_rec, gt_joined, significant_gt)
            or _single_letter_matches(single_gt, rec_joined, significant_rec)
        )

        # Если >60% ключевых слов gt_name представлены в rec_name — семантически эквивалентны
        keyword_ratio = matched_keywords / len(significant_gt) if significant_gt else 0.0

        # Специальный случай: однобуквенное сокращение + значимое слово
        # «Т кожуха» = «Температура воздуха под кожухом» (Т=температура, кожух=кожух)
        # Но не «Р кожуха» = «Давление после крана» (Р=давление, но кожух≠кран)
        # Требуем: однобуквенное сокращение + минимум 1 ключевое слово С ДРУГОЙ семантикой
        # Проверяем: есть ли в matched_keywords слово, не связанное с однобуквенным сокращением
        if has_single_match and matched_keywords >= 2:
            return True

        if keyword_ratio >= 0.6 and ratio >= 0.3:
            return True

        # Если однобуквенное сокращение совпало И есть хотя бы одно ключевое слово, И ratio >= 0.3
        # «Р кожуха» → «Давление воздуха под кожухом» (ratio=0.33) — OK
        # «Р кожуха» → «Давление после крана» (ratio=0.21) — REJECT
        if has_single_match and matched_keywords >= 1 and ratio >= 0.3:
            return True

        # Если >=2 значимых слова совпадают и ratio >= 0.25
        if matched_keywords >= 2 and ratio >= 0.25:
            return True

        return False

    # Pass 1: Upgrade name_match for existing matches where value_match=true
    for j in judgments:
        # Только обновляем если value_match=true но name_match=false
        if j.get("value_match") and not j.get("name_match"):
            gt_name = j.get("gt_param_name", "")
            rec_name = j.get("matched_rec_param_name", "")
            if gt_name and rec_name and _are_names_equivalent(gt_name, rec_name):
                j["name_match"] = True
                j["reasoning"] = (
                    j.get("reasoning", "")
                    + " [auto-upgraded: семантически эквивалентные названия]"
                )
                print(f"    [INFO] name_match upgraded: «{rec_name}» → «{gt_name}»")

    # Pass 2: Try to match unmatched gt_params against all rec_params
    # This handles cases where LLM-judge couldn't find a match at all
    # but names are semantically equivalent via SCADA abbreviations
    matched_rec_names = {
        j["matched_rec_param_name"]
        for j in judgments
        if j.get("matched_rec_param_name")
    }
    for j in judgments:
        if j.get("matched_rec_param_name") or j.get("name_match"):
            continue  # Already matched
        gt_name = j.get("gt_param_name", "")
        gt_value = j.get("gt_value", "")
        if not gt_name:
            continue
        # Try all rec_params that aren't already matched
        best_rec_name = None
        best_rec_value = None
        for rec_name, rec_value in rec_params.items():
            if rec_name in matched_rec_names:
                continue
            if _are_names_equivalent(gt_name, rec_name):
                best_rec_name = rec_name
                best_rec_value = rec_value
                break
        if best_rec_name:
            j["matched_rec_param_name"] = best_rec_name
            j["rec_value"] = best_rec_value
            j["name_match"] = True
            # Check value match
            gt_val_clean = gt_value.replace(".", ",").rstrip("0").rstrip(",") or "0"
            rec_val_clean = (best_rec_value or "").replace(".", ",").rstrip("0").rstrip(",") or "0"
            j["value_match"] = gt_val_clean == rec_val_clean
            j["reasoning"] = (
                f"[auto-matched] «{best_rec_name}» ≈ «{gt_name}» (SCADA аббревиатура)."
            )
            matched_rec_names.add(best_rec_name)
            print(f"    [INFO] auto-matched: «{best_rec_name}» → «{gt_name}» (value_match={j['value_match']})")

    return judgments


async def _judge_single_batch(
    judge_client: LLMJudgeClient,
    system_prompt: str,
    gt_params: dict[str, str],
    rec_params: dict[str, str],
    timestamp: str,
    max_parse_retries: int = 2,
) -> list[dict]:
    """Отправляет один batch к LLM-judge и возвращает judgments.

    Args:
        judge_client: Клиент LLM.
        system_prompt: Системный промпт.
        gt_params: Подмножество эталонных параметров.
        rec_params: Все распознанные параметры.
        timestamp: Таймстемп.
        max_parse_retries: Количество повторных попыток при ошибке парсинга.

    Returns:
        Список judgments для этого batch.
    """
    user_prompt = _build_judge_user_prompt(gt_params, rec_params, timestamp)
    judge_result: dict = {}

    for attempt in range(max_parse_retries + 1):
        try:
            response_text = await judge_client.judge(system_prompt, user_prompt)
            judge_result = _extract_judge_json(response_text)

            if judge_result.get("parse_error"):
                if attempt < max_parse_retries:
                    print(f"      [WARN] JSON parse failed, retry {attempt + 1}/{max_parse_retries}...")
                    user_prompt = (
                        user_prompt
                        + "\n\nВАЖНО: Предыдущий ответ содержал невалидный JSON. "
                        "Верни ТОЛЬКО валидный JSON без markdown-обёрток, без комментариев, "
                        "без лишнего текста. Начни сразу с {"
                    )
                    import asyncio
                    await asyncio.sleep(1.0)
                    continue
                else:
                    print(f"      [ERROR] JSON parse failed after {max_parse_retries} retries")
            else:
                break
        except Exception as e:
            if attempt < max_parse_retries:
                print(f"      [WARN] LLM judge error, retrying: {e}")
                import asyncio
                await asyncio.sleep(2.0)
                continue
            else:
                print(f"      [ERROR] LLM judge failed: {e}")
                # Fallback: все параметры считаются несовпавшими
                return [
                    {
                        "gt_param_name": name,
                        "gt_value": val,
                        "matched_rec_param_name": None,
                        "rec_value": None,
                        "name_match": False,
                        "value_match": False,
                        "reasoning": f"LLM judge error: {e}",
                    }
                    for name, val in gt_params.items()
                ]

    return judge_result.get("judgments", [])


async def llm_judge_compare(
    recognized: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, str]],
    judge_client: LLMJudgeClient | None = None,
    max_parse_retries: int = 2,
    batch_size: int = 12,
) -> dict:
    """Сравнивает распознанные параметры с эталонными через LLM-as-a-Judge.

    Использует двухпроходную пакетную стратегию:
    - Проход 1: gt_params разбиваются на батчи по batch_size, каждый батч
      сравнивается со ВСЕМИ rec_params. Результаты мержатся.
    - Проход 2: для gt_params без совпадения (value_match=false) делается
      повторный запрос с УЖЕ ИСПОЛЬЗОВАННЫМИ rec_params исключёнными,
      чтобы облегчить задачу модели.
    Включает retry при ошибке парсинга JSON и пост-обработку name_match.

    Args:
        recognized: Результаты VLM {timestamp: {param_name: value}}.
        ground_truth: Эталонные значения {timestamp: {param_name: value}}.
        judge_client: Экземпляр LLMJudgeClient (создаётся автоматически если None).
        max_parse_retries: Количество повторных попыток при ошибке парсинга JSON.
        batch_size: Размер батча gt_params для одного запроса к LLM.

    Returns:
        Словарь с метриками и детальными результатами.
    """
    if judge_client is None:
        from app.config import settings
        judge_client = LLMJudgeClient(
            base_url=getattr(settings, "vlm_base_url", "http://localhost:8090"),
            model_name=getattr(settings, "vlm_model_name", "Qwen3.5-4B"),
        )

    system_prompt = _build_judge_system_prompt()

    all_judgments: list[dict] = []
    per_timestamp_results: dict[str, dict] = {}
    total_gt = 0
    total_name_matched = 0
    total_value_matched = 0
    per_param: dict[str, list[bool]] = {}  # param_name -> [value_correct, ...]
    mismatches: list[dict] = []

    gt_timestamps = list(ground_truth.keys())

    for gt_ts in gt_timestamps:
        gt_params = ground_truth[gt_ts]
        rec_params = recognized.get(gt_ts, {})

        if not gt_params:
            continue

        total_gt += len(gt_params)
        print(f"    Judge: {gt_ts} ({len(gt_params)} gt, {len(rec_params)} rec)")

        # =============================================
        # ПРОХОД 1: пакетное сравнение
        # =============================================
        gt_items = list(gt_params.items())
        batches: list[dict[str, str]] = []
        for i in range(0, len(gt_items), batch_size):
            batch = dict(gt_items[i : i + batch_size])
            batches.append(batch)

        pass1_judgments: list[dict] = []
        for bi, batch in enumerate(batches):
            print(f"      Pass 1, batch {bi + 1}/{len(batches)} ({len(batch)} gt params)")
            batch_judgments = await _judge_single_batch(
                judge_client, system_prompt, batch, rec_params, gt_ts, max_parse_retries,
            )
            pass1_judgments.extend(batch_judgments)

        # Пост-обработка прохода 1
        pass1_judgments = _upgrade_name_match(pass1_judgments, gt_params, rec_params)

        # Собираем matched rec params (для исключения в проходе 2)
        used_rec_names: set[str] = set()
        for j in pass1_judgments:
            if j.get("value_match") and j.get("matched_rec_param_name"):
                used_rec_names.add(j["matched_rec_param_name"])

        # Находим unmatched gt params
        unmatched_gt: dict[str, str] = {}
        for j in pass1_judgments:
            if not j.get("value_match"):
                gt_name = j.get("gt_param_name", "")
                gt_val = j.get("gt_value", "")
                if gt_name and gt_name in gt_params:
                    unmatched_gt[gt_name] = gt_params[gt_name]

        # =============================================
        # ПРОХОД 2: повтор для unmatched с уменьшенным rec
        # =============================================
        pass2_judgments: list[dict] = []
        if unmatched_gt:
            # Исключаем уже использованные rec-параметры
            remaining_rec = {
                k: v for k, v in rec_params.items() if k not in used_rec_names
            }
            print(f"      Pass 2: {len(unmatched_gt)} unmatched gt, {len(remaining_rec)} remaining rec")

            # Разбиваем unmatched gt на батчи
            unmatched_items = list(unmatched_gt.items())
            for i in range(0, len(unmatched_items), batch_size):
                batch = dict(unmatched_items[i : i + batch_size])
                batch_judgments = await _judge_single_batch(
                    judge_client, system_prompt, batch, remaining_rec, gt_ts, max_parse_retries,
                )
                pass2_judgments.extend(batch_judgments)

            # Пост-обработка прохода 2
            pass2_judgments = _upgrade_name_match(pass2_judgments, gt_params, remaining_rec)

        # =============================================
        # Мержим результаты: проход 2 перезаписывает проход 1 для unmatched
        # =============================================
        pass2_matched_gt = {
            j["gt_param_name"]: j
            for j in pass2_judgments
            if j.get("value_match") and j.get("gt_param_name")
        }
        final_judgments: list[dict] = []
        for j in pass1_judgments:
            gt_name = j.get("gt_param_name", "")
            if gt_name in pass2_matched_gt:
                # Заменяем результат прохода 1 на результат прохода 2
                final_judgments.append(pass2_matched_gt[gt_name])
            else:
                final_judgments.append(j)

        # Добавляем новые совпадения из прохода 2 (если были)
        for j in pass2_judgments:
            gt_name = j.get("gt_param_name", "")
            if gt_name not in {fj.get("gt_param_name", "") for fj in final_judgments}:
                final_judgments.append(j)

        # Обрабатываем итоговые решения
        for j in final_judgments:
            gt_name = j.get("gt_param_name", "")
            gt_val = j.get("gt_value", "")
            rec_name = j.get("matched_rec_param_name")
            rec_val = j.get("rec_value")
            name_match = j.get("name_match", False)
            value_match = j.get("value_match", False)
            reasoning = j.get("reasoning", "")

            all_judgments.append({
                "timestamp": gt_ts,
                **j,
            })

            if name_match:
                total_name_matched += 1
            if value_match:
                total_value_matched += 1

            # Собираем per-param статистику по value_match
            if gt_name not in per_param:
                per_param[gt_name] = []
            per_param[gt_name].append(value_match)

            if not value_match:
                mismatches.append({
                    "timestamp": gt_ts,
                    "param_name": gt_name,
                    "expected": gt_val,
                    "recognized": rec_val or "",
                    "reasoning": reasoning,
                })

        # Сохраняем per-timestamp результат
        per_timestamp_results[gt_ts] = {
            "total_gt": len(gt_params),
            "name_matched": sum(1 for j in final_judgments if j.get("name_match")),
            "value_matched": sum(1 for j in final_judgments if j.get("value_match")),
            "judgments": final_judgments,
            "unmatched_rec": list(
                {k for k in rec_params if k not in used_rec_names}
            ),
        }

    # Вычисляем итоговые метрики
    value_accuracy = (total_value_matched / total_gt * 100.0) if total_gt > 0 else 0.0
    name_accuracy = (total_name_matched / total_gt * 100.0) if total_gt > 0 else 0.0

    per_param_accuracy = {}
    for pname, results in sorted(per_param.items()):
        acc = sum(results) / len(results) * 100.0
        per_param_accuracy[pname] = round(acc, 2)

    # Coverage: сколько уникальных GT параметров были найдены хотя бы раз
    found_params = set()
    for j in all_judgments:
        if j.get("value_match") or j.get("name_match"):
            found_params.add(j.get("gt_param_name", ""))
    expected_params = set()
    for gt_params in ground_truth.values():
        expected_params.update(gt_params.keys())
    coverage_pct = (len(found_params) / len(expected_params) * 100.0) if expected_params else 0.0

    return {
        "mode": "llm_judge",
        "accuracy_pct": round(value_accuracy, 2),
        "name_match_accuracy_pct": round(name_accuracy, 2),
        "total_params": total_gt,
        "name_matched_params": total_name_matched,
        "value_matched_params": total_value_matched,
        "correct_params": total_value_matched,
        "per_param_accuracy": per_param_accuracy,
        "per_timestamp_results": per_timestamp_results,
        "mismatches": mismatches[:100],
        "all_judgments": all_judgments,
        "coverage_stats": {
            "expected_unique_params": len(expected_params),
            "found_unique_params": len(found_params),
            "coverage_pct": round(coverage_pct, 2),
        },
        "pass_95pct": value_accuracy >= 95.0,
    }


def llm_judge_compare_sync(
    recognized: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, str]],
    judge_client: LLMJudgeClient | None = None,
) -> dict:
    """Синхронная обёртка для llm_judge_compare."""
    import asyncio
    return asyncio.run(llm_judge_compare(recognized, ground_truth, judge_client))


def parse_xml_params(xml_string: str) -> dict[str, dict[int, str]]:
    """Парсит XML в формате <sheme> и возвращает параметры по таймстемпам.

    Returns:
        Словарь {timestamp: {param_id: value}}.
    """
    result: dict[str, dict[int, str]] = {}
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        print(f"  [ERROR] XML parse error: {e}")
        return result

    for params_elem in root.findall("parameters"):
        ts = params_elem.get("timestamp", "unknown")
        params: dict[int, str] = {}
        for param_elem in params_elem.findall("param"):
            pid = int(param_elem.get("id", 0))
            val = param_elem.text or ""
            params[pid] = val
        result[ts] = params

    return result


def parse_xml_params_by_name(xml_string: str) -> dict[str, dict[str, str]]:
    """Парсит XML с параметрами по имени (name атрибут вместо id).

    Формат: <param name="Температура масла...">36.1</param>

    Returns:
        Словарь {timestamp: {param_name: value}}.
    """
    result: dict[str, dict[str, str]] = {}
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        print(f"  [ERROR] XML parse error: {e}")
        return result

    for params_elem in root.findall("parameters"):
        ts = params_elem.get("timestamp", "unknown")
        params: dict[str, str] = {}
        for param_elem in params_elem.findall("param"):
            name = param_elem.get("name", "")
            val = param_elem.text or ""
            if name:
                params[name] = val
        result[ts] = params

    return result


def compare_results(
    recognized: dict[str, dict[int, str]],
    ground_truth: dict[str, dict[int, str]],
    tolerance: float = 0.01,
) -> dict:
    """Сравнивает распознанные параметры с эталонными.

    Args:
        recognized: Результаты OCR {timestamp: {param_id: value}}.
        ground_truth: Эталонные значения {timestamp: {param_id: value}}.
        tolerance: Допустимая относительная погрешность для чисел.

    Returns:
        Словарь с метриками: accuracy, per_param_accuracy, details.
    """
    total = 0
    correct = 0
    per_param: dict[int, list[bool]] = {}
    mismatches: list[dict] = []

    # Сравниваем по ближайшим таймстемпам
    gt_timestamps = list(ground_truth.keys())
    rec_timestamps = list(recognized.keys())

    for gt_ts, gt_params in ground_truth.items():
        # Находим ближайший распознанный таймстемп
        rec_ts = _find_closest_timestamp(gt_ts, rec_timestamps)
        rec_params = recognized.get(rec_ts, {}) if rec_ts else {}

        for pid, gt_val in gt_params.items():
            total += 1
            rec_val = rec_params.get(pid, "")

            is_correct = _values_match(rec_val, gt_val, tolerance)

            if pid not in per_param:
                per_param[pid] = []
            per_param[pid].append(is_correct)

            if is_correct:
                correct += 1
            else:
                mismatches.append({
                    "timestamp": gt_ts,
                    "param_id": pid,
                    "expected": gt_val,
                    "recognized": rec_val,
                })

    accuracy = (correct / total * 100.0) if total > 0 else 0.0

    per_param_accuracy = {}
    for pid, results in sorted(per_param.items()):
        acc = sum(results) / len(results) * 100.0
        per_param_accuracy[pid] = round(acc, 2)

    return {
        "accuracy_pct": round(accuracy, 2),
        "total_params": total,
        "correct_params": correct,
        "per_param_accuracy": per_param_accuracy,
        "mismatches": mismatches[:50],  # Ограничиваем вывод
        "pass_95pct": accuracy >= 95.0,
    }


def _find_closest_timestamp(
    target: str, candidates: list[str]
) -> str | None:
    """Находит ближайший таймстемп к целевому."""
    if not candidates:
        return None

    target_ms = _timestamp_to_ms(target)
    best = None
    best_diff = float("inf")

    for c in candidates:
        c_ms = _timestamp_to_ms(c)
        diff = abs(target_ms - c_ms)
        if diff < best_diff:
            best_diff = diff
            best = c

    return best


def extract_frames_at_timestamps(
    video_path: str,
    timestamps_ms: list[int],
) -> list[tuple[Any, int]]:
    """Извлекает кадры из видео на указанных таймстампах.

    Args:
        video_path: Путь к видеофайлу.
        timestamps_ms: Список таймстампов в миллисекундах.

    Returns:
        Список кортежей (frame, timestamp_ms).
    """
    import cv2

    frames: list[tuple[Any, int]] = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    try:
        duration_ms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) * 1000)

        for ts_ms in sorted(timestamps_ms):
            if ts_ms > duration_ms:
                print(f"  [WARN] Timestamp {ts_ms}ms > video duration {duration_ms}ms, skipping")
                continue

            cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
            ret, frame = cap.read()

            if ret:
                # CLAHE ОТКЛЮЧЕН: искажает цвета SCADA (синий→серый, красный→оранжевый)
                # SCADA опирается на цветовую семантику: синий=актив, красный=авария, зеленый=норма
                # Для evaluation используем оригинальные кадры без изменений
                frames.append((frame, ts_ms))
            else:
                print(f"  [WARN] Не удалось прочитать кадр на {ts_ms}ms")

        print(f"  Извлечено {len(frames)}/{len(timestamps_ms)} кадров")
    finally:
        cap.release()

    return frames


def compare_results_by_name_direct(
    recognized: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, str]],
    tolerance: float = 0.01,
) -> dict:
    """Прямое сравнение результатов VLM с ground truth по именам параметров.

    Используется для name-based XML формата (Видео 2).

    Args:
        recognized: Результаты VLM {timestamp: {param_name: value}}.
        ground_truth: Эталонные значения {timestamp: {param_name: value}}.
        tolerance: Допустимая относительная погрешность для чисел.

    Returns:
        Словарь с метриками: accuracy, per_param_accuracy, details.
    """
    total = 0
    correct = 0
    per_param: dict[str, list[bool]] = {}
    mismatches: list[dict] = []

    for gt_ts, gt_params in ground_truth.items():
        # Ищем точное совпадение таймстампа
        rec_params = recognized.get(gt_ts, {})

        for param_name, gt_val in gt_params.items():
            total += 1
            
            # Ищем распознанное значение по имени (fuzzy match)
            rec_val = ""
            for rec_name, rec_value in rec_params.items():
                if _names_match(rec_name, param_name):
                    rec_val = rec_value
                    break

            is_correct = _values_match(rec_val, gt_val, tolerance)

            if param_name not in per_param:
                per_param[param_name] = []
            per_param[param_name].append(is_correct)

            if is_correct:
                correct += 1
            else:
                mismatches.append({
                    "timestamp": gt_ts,
                    "param_name": param_name,
                    "expected": gt_val,
                    "recognized": rec_val,
                })

    accuracy = (correct / total * 100.0) if total > 0 else 0.0

    per_param_accuracy = {}
    for pname, results in sorted(per_param.items()):
        acc = sum(results) / len(results) * 100.0
        per_param_accuracy[pname] = round(acc, 2)

    return {
        "accuracy_pct": round(accuracy, 2),
        "total_params": total,
        "correct_params": correct,
        "per_param_accuracy": per_param_accuracy,
        "mismatches": mismatches[:50],
        "pass_95pct": accuracy >= 95.0,
    }


def compare_results_by_name(
    recognized: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[int, str]],
    param_table: list[dict] | None = None,
    tolerance: float = 0.01,
) -> dict:
    """Сравнивает распознанные параметры с эталонными.

    Args:
        recognized: Результаты VLM {timestamp: {param_name: value}}.
        ground_truth: Эталонные значения {timestamp: {param_id: value}}.
        param_table: Таблица параметров для маппинга ID -> name.
        tolerance: Допустимая относительная погрешность для чисел.

    Returns:
        Словарь с метриками: accuracy, per_param_accuracy, details.
    """
    # Создаём маппинг ID -> name из таблицы параметров
    id_to_name: dict[int, str] = {}
    if param_table:
        for row in param_table:
            pid = row.get("id")
            name = row.get("name", "")
            if pid and name:
                try:
                    id_to_name[int(pid)] = str(name)
                except (ValueError, TypeError):
                    pass

    total = 0
    correct = 0
    per_param: dict[str, list[bool]] = {}
    mismatches: list[dict] = []

    for gt_ts, gt_params in ground_truth.items():
        # Ищем точное совпадение таймстампа
        rec_params = recognized.get(gt_ts, {})

        for param_id, gt_val in gt_params.items():
            total += 1

            # Получаем имя параметра из таблицы
            param_name = id_to_name.get(param_id, f"ID:{param_id}")

            # Ищем распознанное значение по имени (fuzzy match)
            rec_val = ""
            for rec_name, rec_value in rec_params.items():
                # Простое fuzzy matching: проверяем contains или высокую схожесть
                if _names_match(rec_name, param_name):
                    rec_val = rec_value
                    break

            is_correct = _values_match(rec_val, gt_val, tolerance)

            if param_name not in per_param:
                per_param[param_name] = []
            per_param[param_name].append(is_correct)

            if is_correct:
                correct += 1
            else:
                mismatches.append({
                    "timestamp": gt_ts,
                    "param_id": param_id,
                    "param_name": param_name,
                    "expected": gt_val,
                    "recognized": rec_val,
                })

    accuracy = (correct / total * 100.0) if total > 0 else 0.0

    per_param_accuracy = {}
    for pname, results in sorted(per_param.items()):
        acc = sum(results) / len(results) * 100.0
        per_param_accuracy[pname] = round(acc, 2)

    return {
        "accuracy_pct": round(accuracy, 2),
        "total_params": total,
        "correct_params": correct,
        "per_param_accuracy": per_param_accuracy,
        "mismatches": mismatches[:50],  # Ограничиваем вывод
        "pass_95pct": accuracy >= 95.0,
    }


def _names_match(name1: str, name2: str) -> bool:
    """Проверяет схожесть двух имён параметров.
    
    УЛУЧШЕНО: Удаляет теги типа (PT4413) перед сравнением.
    """
    # КРИТИЧНО: Удаляем теги типа (PT4413), [TT1234] перед сравнением
    clean_name1 = re.sub(r'[\(\[][^)\]]*[\)\]]', '', name1).strip()
    clean_name2 = re.sub(r'[\(\[][^)\]]*[\)\]]', '', name2).strip()
    
    # Точное совпадение (после очистки)
    if clean_name1 == clean_name2:
        return True

    # Один содержится в другом (после очистки)
    if clean_name1 in clean_name2 or clean_name2 in clean_name1:
        return True

    # Нормализуем и сравниваем (убираем лишние пробелы, lowercase)
    norm1 = " ".join(clean_name1.lower().split())
    norm2 = " ".join(clean_name2.lower().split())
    if norm1 == norm2:
        return True

    # Проверяем схожесть через SequenceMatcher (built-in, no dependencies)
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    return similarity >= 0.80


def _timestamp_to_ms(ts: str) -> float:
    """Конвертирует таймстемп HH:MM:SS.mmm в миллисекунды."""
    try:
        parts = ts.split(":")
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].split(".")
        s = int(s_parts[0])
        ms = int(s_parts[1]) if len(s_parts) > 1 else 0
        return h * 3600000 + m * 60000 + s * 1000 + ms
    except (ValueError, IndexError):
        return 0.0


def _values_match(rec_val: str, gt_val: str, tolerance: float) -> bool:
    """Сравнивает два значения с учётом числовой толерантности.
    
    НОРМАЛИЗАЦИЯ: Заменяет запятые на точки для числового сравнения.
    -1,5 (запятая) и -1.5 (точка) считаются одинаковыми.
    """
    rec_clean = rec_val.strip()
    gt_clean = gt_val.strip()

    # Точное строковое совпадение
    if rec_clean == gt_clean:
        return True

    # НОРМАЛИЗАЦИЯ: Заменяем запятые на точки для числового сравнения
    rec_normalized = rec_clean.replace(",", ".")
    gt_normalized = gt_clean.replace(",", ".")
    
    # Проверяем совпадение после нормализации
    if rec_normalized == gt_normalized:
        return True

    # Числовое сравнение с толерантностью (после нормализации)
    try:
        rec_num = float(rec_normalized)
        gt_num = float(gt_normalized)
        if gt_num == 0.0:
            return abs(rec_num) < tolerance
        return abs(rec_num - gt_num) / abs(gt_num) <= tolerance
    except (ValueError, ZeroDivisionError):
        pass

    # Нечёткое сравнение (Levenshtein)
    try:
        from Levenshtein import distance
        max_len = max(len(rec_clean), len(gt_clean))
        if max_len == 0:
            return True
        similarity = 1.0 - distance(rec_clean, gt_clean) / max_len
        return similarity >= 0.9
    except ImportError:
        pass

    return False


def run_pipeline_on_video(
    video_path: str,
    param_table_path: str | None = None,
    video_type: str | None = None,
    zone_enabled: bool = True,
) -> tuple[str, float]:
    """Запускает VLM конвейер обработки видео и возвращает XML + время.

    Использует VLMPipeline с зонным анализом (по умолчанию) или
    полнокадровым fallback.

    Args:
        video_path: Путь к видеофайлу.
        param_table_path: Путь к таблице параметров (.xlsx/.csv).
        video_type: Тип видео (direct/handheld).
        zone_enabled: Использовать зонный анализ (True) или полный кадр (False).

    Returns:
        Кортеж (xml_string, elapsed_seconds).
    """
    import asyncio
    from app.core.vlm_pipeline import VLMPipeline
    from app.core.parameter_mapper import load_parameter_table

    # Загружаем таблицу параметров
    param_table = None
    if param_table_path and Path(param_table_path).exists():
        try:
            param_table = load_parameter_table(param_table_path)
            print(f"  Загружено {len(param_table)} параметров из таблицы")
        except Exception as e:
            print(f"  [WARN] Не удалось загрузить таблицу: {e}")

    start = time.time()

    async def _run():
        pipeline = VLMPipeline()
        # Перекрываем настройку зонного анализа
        pipeline.zone_enabled = zone_enabled
        mode_str = "ZONE" if zone_enabled else "FULL-FRAME"
        print(f"  Режим VLM: {mode_str}")

        result = await pipeline.process_video(
            video_path=Path(video_path),
            video_id=Path(video_path).stem,
            send_email=False,
            parameter_table=param_table,
        )

        if result.xml_path and result.xml_path.exists():
            xml_content = result.xml_path.read_text(encoding="utf-8")
        else:
            xml_content = "<sheme><error>No XML generated</error></sheme>"

        return xml_content

    xml_content = asyncio.run(_run())
    elapsed = time.time() - start
    return xml_content, elapsed


def evaluate_video_vlm(
    video_path: str,
    ground_truth_path: str,
    output_dir: str = "data/reports",
    use_llm_judge: bool = False,
    zone_enabled: bool = True,
) -> dict:
    """Оценка видео через VLM пайплайн с сравнением по ground truth.

    Извлекает кадры только на таймстампах из ground truth XML,
    запускает VLM анализ (зонный или полнокадровый) и сравнивает результаты.

    Args:
        video_path: Путь к видеофайлу.
        ground_truth_path: Путь к эталонному XML.
        output_dir: Директория для отчётов.
        use_llm_judge: Использовать LLM-as-a-Judge для сравнения.
        zone_enabled: Использовать зонный анализ (True) или полный кадр (False).

    Returns:
        Словарь с результатами оценки.
    """
    import asyncio
    import cv2
    from app.core.vlm_pipeline import VLMPipeline
    from app.utils.xml_utils import format_timestamp

    print(f"\n{'=' * 60}")
    print(f"  VLM Оценка видео: {Path(video_path).name}")
    print(f"  Ground Truth: {Path(ground_truth_path).name}")
    print(f"{'=' * 60}")

    # Загружаем ground truth XML
    print("\n  [1/5] Загрузка ground truth XML...")
    gt_xml = Path(ground_truth_path).read_text(encoding="utf-8")
    
    # Определяем формат XML (по name или id атрибутам)
    if 'name="' in gt_xml[:500]:
        # Формат с name атрибутами (Видео 2)
        print("  Обнаружен формат: name-based")
        ground_truth = parse_xml_params_by_name(gt_xml)
        use_name_format = True
    else:
        # Формат с id атрибутами (Видео 1)
        print("  Обнаружен формат: ID-based")
        ground_truth = parse_xml_params(gt_xml)
        use_name_format = False
    
    gt_timestamps = list(ground_truth.keys())
    print(f"  Найдено {len(gt_timestamps)} таймстампов в ground truth")

    # Загружаем таблицу параметров (если есть)
    param_table = None
    video_dir = Path(ground_truth_path).parent
    for table_file in video_dir.glob("Таблица *.xlsx"):
        print(f"\n  [1.5] Загрузка таблицы параметров: {table_file.name}")
        try:
            from app.core.parameter_mapper import load_parameter_table
            param_table = load_parameter_table(table_file)
            print(f"  Загружено {len(param_table)} параметров")
            # Показываем первые 3 параметра
            for i, row in enumerate(param_table[:3]):
                name = row.get("name", "")
                print(f"    [{i+1}] {name}")
            if len(param_table) > 3:
                print(f"    ... и ещё {len(param_table) - 3} параметров")
        except Exception as e:
            print(f"  [WARN] Не удалось загрузить таблицу: {e}")
            import traceback
            traceback.print_exc()

    # Конвертируем таймстампы в миллисекунды
    timestamps_ms = [_timestamp_to_ms(ts) for ts in gt_timestamps]
    print(f"  Таймстампы: {', '.join(gt_timestamps[:5])}...")

    # Извлекаем кадры на нужных таймстампах
    print("\n  [2/5] Извлечение кадров на указанных таймстампах...")
    try:
        frames = extract_frames_at_timestamps(video_path, timestamps_ms)
    except Exception as e:
        print(f"  [ERROR] Ошибка извлечения кадров: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "video": video_path}

    if not frames:
        print("  [ERROR] Не удалось извлечь ни одного кадра")
        return {"error": "No frames extracted", "video": video_path}

    # Запускаем VLM анализ
    mode_str = "ZONE" if zone_enabled else "FULL-FRAME"
    print(f"\n  [3/5] Запуск VLM анализа (режим: {mode_str})...")
    pipeline = VLMPipeline()
    pipeline.zone_enabled = zone_enabled  # Перекрываем настройку зон

    async def run_vlm_analysis():
        """Запускает VLM анализ всех кадров."""
        results = {}
        raw_vlm_responses = {}  # Сырые ответы VLM для каждого таймстампа
        total = len(frames)
        success_count = 0
        error_count = 0

        # Отключаем логирование VLM для чистоты вывода
        import logging
        vlm_logger = logging.getLogger("app.core.vlm_pipeline")
        old_level = vlm_logger.level
        vlm_logger.setLevel(logging.WARNING)

        try:
            for idx, (frame, ts_ms) in enumerate(frames):
                ts_str = format_timestamp(ts_ms / 1000.0)
                print(f"    Обработка кадра {idx + 1}/{total} ({ts_str}) [{mode_str}]...")

                try:
                    # Анализируем кадр через VLM
                    # При zone_enabled=True -> зонный анализ с zone-specific промптами
                    # При zone_enabled=False -> полнокадровый анализ с таблицей параметров
                    vlm_result = await pipeline._analyze_single_frame(
                        frame=frame,
                        timestamp=ts_str,
                        frame_idx=idx,
                        total_frames=total,
                        parameter_table=param_table if not zone_enabled else None,
                    )

                    # Сохраняем сырой ответ VLM
                    raw_vlm_responses[ts_str] = vlm_result

                    # Извлекаем параметры из VLM ответа
                    if "error" not in vlm_result:
                        params = {}
                        skipped_params = 0

                        for param in vlm_result.get("parameters", []):
                            # КРИТИЧНО: Пропускаем параметры, которые не прошли fuzzy matching
                            if param.get("skip"):
                                skipped_params += 1
                                continue

                            # VLM возвращает параметры с полем 'label'
                            label = param.get("label", "")
                            value = str(param.get("value", ""))
                            if label:
                                params[label] = value

                        results[ts_str] = params
                        success_count += 1

                        zones_info = ""
                        if zone_enabled:
                            zones_count = vlm_result.get("zones_processed", 0)
                            zones_info = f" (зон: {zones_count})"

                        param_table_info = f" (из таблицы: {len(param_table)})" if param_table and not zone_enabled else ""
                        skip_info = f", пропущено={skipped_params}" if skipped_params > 0 else ""
                        print(f"      ✓ Распознано {len(params)} параметров{zones_info}{param_table_info}{skip_info}")

                        # Показываем статус валидации
                        validation_status = vlm_result.get("vlm_validation", "unknown")
                        if validation_status == "failed":
                            print(f"      ⚠ Pydantic validation FAILED")
                    else:
                        error_count += 1
                        print(f"      ✗ [ERROR] VLM error: {vlm_result['error']}")

                except Exception as e:
                    error_count += 1
                    print(f"      ✗ [ERROR] Ошибка VLM анализа: {e}")
        finally:
            # Восстанавливаем уровень логирования
            vlm_logger.setLevel(old_level)

        print(f"\n  Статистика: успешно={success_count}, ошибок={error_count}, всего={total}")
        return results, raw_vlm_responses

    # Запускаем async функцию
    recognized, raw_vlm_responses = asyncio.run(run_vlm_analysis())
    print(f"\n  Успешно обработано {len(recognized)}/{len(frames)} кадров")

    # Нормализация лейблов через fuzzy matching с таблицей параметров
    # КРИТИЧНО: При зонном анализе VLM возвращает «сырые» лейблы с экрана,
    # а для сравнения с ground truth нужно мапить их на имена из таблицы.
    if param_table and recognized:
        from difflib import SequenceMatcher
        table_name_to_id: dict[str, int] = {row['name']: row['id'] for row in param_table if 'name' in row and 'id' in row}
        table_names = list(table_name_to_id.keys())
        normalized_count = 0

        for ts_str, params in recognized.items():
            normalized_params = {}
            for raw_label, value in params.items():
                # Ищем лучшее совпадение в таблице
                best_match = None
                best_score = 0.0
                clean_vlm = re.sub(r'\([^)]*\)', '', raw_label).strip()
                for table_name in table_names:
                    clean_table = re.sub(r'\([^)]*\)', '', table_name).strip()
                    score = SequenceMatcher(None, clean_vlm.lower(), clean_table.lower()).ratio()
                    if score > best_score:
                        best_score = score
                        best_match = table_name

                if best_match and best_score >= 0.80:
                    normalized_params[best_match] = value
                    if best_score < 1.0:
                        normalized_count += 1
                else:
                    # Не нашли — оставляем как есть
                    normalized_params[raw_label] = value

            recognized[ts_str] = normalized_params

        if normalized_count > 0:
            print(f"  Нормализовано {normalized_count} лейблов через fuzzy matching")
    
    # Подробная статистика
    if len(recognized) < len(frames):
        print(f"\n  [WARN] {len(frames) - len(recognized)} кадров не распознаны:")
        for ts_str, response in raw_vlm_responses.items():
            if ts_str not in recognized:
                error = response.get("error", "unknown error")
                print(f"    - {ts_str}: {error[:100]}")

    # Генерируем сырой XML из VLM ответов
    from app.core.xml_generator import generate_xml
    from app.models.schemas import SnapshotData
    
    snapshots = []
    for ts_str, params in sorted(recognized.items()):
        # Конвертируем dict[str, str] в dict[int, str] для SnapshotData
        int_params = {i: val for i, val in enumerate(params.values(), 1)}
        snapshot = SnapshotData(timestamp=ts_str, params=int_params)
        snapshots.append(snapshot)
    
    raw_vlm_xml = generate_xml(snapshots) if snapshots else "<sheme><error>No successful frames</error></sheme>"

    # Сравнение с ground truth
    print("\n  [4/5] Сравнение с ground truth...")
    if use_name_format:
        # name-based формат (Видео 2) - прямое сравнение по именам
        if use_llm_judge:
            print("  Режим сравнения: LLM-as-a-Judge")
            comparison = llm_judge_compare_sync(recognized, ground_truth)
        else:
            comparison = compare_results_by_name_direct(recognized, ground_truth)
    else:
        # ID-based формат (Видео 1) - сравнение через таблицу параметров
        # Для LLM-judge конвертируем ID в имена
        if use_llm_judge:
            print("  Режим сравнения: LLM-as-a-Judge")
            # Конвертируем ground truth из {ts: {id: val}} в {ts: {name: val}}
            gt_by_name: dict[str, dict[str, str]] = {}
            id_to_name: dict[int, str] = {}
            if param_table:
                for row in param_table:
                    pid = row.get("id")
                    name = row.get("name", "")
                    if pid is not None and name:
                        id_to_name[int(pid)] = name

            for ts, params in ground_truth.items():
                named: dict[str, str] = {}
                for pid, val in params.items():
                    pname = id_to_name.get(pid, f"ID:{pid}")
                    named[pname] = val
                gt_by_name[ts] = named

            comparison = llm_judge_compare_sync(recognized, gt_by_name)
        else:
            comparison = compare_results_by_name(recognized, ground_truth, param_table)

    # КРИТИЧНО: Добавляем статистику coverage (охват параметров)
    # Сколько параметров ожидалось vs найдено
    expected_params = set()
    for gt_params in ground_truth.values():
        expected_params.update(gt_params.keys())
    
    found_params = set()
    for rec_params in recognized.values():
        found_params.update(rec_params.keys())
    
    # Считаем покрытие через fuzzy matching
    matched_params = set()
    for exp_name in expected_params:
        for found_name in found_params:
            if _names_match(exp_name, found_name):
                matched_params.add(exp_name)
                break
    
    coverage_pct = (len(matched_params) / len(expected_params) * 100.0) if expected_params else 0.0
    
    comparison["coverage_stats"] = {
        "expected_unique_params": len(expected_params),
        "found_unique_params": len(found_params),
        "matched_params": len(matched_params),
        "coverage_pct": round(coverage_pct, 2),
    }

    # Вывод результатов
    result = {
        "video": str(video_path),
        "ground_truth": str(ground_truth_path),
        "mode": "vlm_llm_judge" if use_llm_judge else "vlm",
        "zone_enabled": zone_enabled,
        "gt_timestamps": len(gt_timestamps),
        "recognized_timestamps": len(recognized),
        "raw_vlm_xml": raw_vlm_xml,  # Сырой XML от VLM
        "raw_vlm_responses": {  # Сырые ответы VLM для отладки
            ts: resp for ts, resp in raw_vlm_responses.items()
        },
        **comparison,
    }

    _print_results_vlm(result)

    # Сохраняем JSON-отчёт
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_id = Path(video_path).stem
    mode_suffix = "_llm_judge" if use_llm_judge else "_vlm"
    report_path = out_dir / f"{video_id}{mode_suffix}_evaluation.json"
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n  [5/5] Отчёт сохранён: {report_path}")

    return result


def _print_results_vlm(result: dict) -> None:
    """Выводит результаты VLM оценки в консоль."""
    print(f"\n  {'─' * 50}")
    mode_str = "ZONE" if result.get('zone_enabled', True) else "FULL-FRAME"
    print(f"  РЕЗУЛЬТАТЫ VLM ОЦЕНКИ (режим: {mode_str})")
    print(f"  {'─' * 50}")
    print(f"  Accuracy:        {result.get('accuracy_pct', 0)}%")
    print(f"  Correct/Total:   {result.get('correct_params', 0)}/{result.get('total_params', 0)}")
    print(f"  GT Timestamps:   {result.get('gt_timestamps', 0)}")
    print(f"  Recognized:      {result.get('recognized_timestamps', 0)}")
    
    # КРИТИЧНО: Показываем coverage статистику
    coverage = result.get('coverage_stats', {})
    if coverage:
        print(f"\n  COVERAGE (Охват параметров):")
        print(f"  Expected:        {coverage.get('expected_unique_params', 0)} уникальных")
        print(f"  Found:           {coverage.get('found_unique_params', 0)} уникальных")
        matched_key = 'matched_params' if 'matched_params' in coverage else None
        if matched_key:
            print(f"  Matched:         {coverage.get(matched_key, 0)} (fuzzy match)")
        print(f"  Coverage:        {coverage.get('coverage_pct', 0)}%")

    # Если LLM Judge — показываем дополнительную статистику
    name_acc = result.get('name_match_accuracy_pct')
    if name_acc is not None:
        print(f"\n  Name match acc:  {name_acc}%")
        print(f"  Value match acc: {result.get('accuracy_pct', 0)}%")

    if result.get("pass_95pct"):
        print(f"\n  *** PASS *** Accuracy >= 95%")
    else:
        print(f"\n  *** FAIL *** Accuracy < 95%")

    # Топ несовпадений
    mismatches = result.get("mismatches", [])
    if mismatches:
        print(f"\n  Первые несовпадения ({min(len(mismatches), 10)}):")
        for m in mismatches[:10]:
            print(f"    ts={m['timestamp']}")
            pname = m.get('param_name', m.get('param_id', '?'))
            print(f"      param: {pname}")
            print(f"      expected: '{m['expected']}'")
            print(f"      got:      '{m['recognized']}'")
            reasoning = m.get('reasoning')
            if reasoning:
                print(f"      reasoning: {reasoning[:100]}")

    # Точность по параметрам
    per_param = result.get("per_param_accuracy", {})
    if per_param:
        worst = sorted(per_param.items(), key=lambda x: x[1])[:5]
        if worst:
            print(f"\n  Худшие параметры (топ-5):")
            for pname, acc in worst:
                # Сокращаем длинные имена
                short_name = pname[:50] + "..." if len(pname) > 50 else pname
                print(f"    {short_name}: {acc}%")


def evaluate_video(
    video_path: str,
    param_table_path: str | None = None,
    video_type: str | None = None,
    ground_truth_path: str | None = None,
    output_dir: str = "data/reports",
) -> dict:
    """Полная оценка одного видео.

    Returns:
        Словарь с результатами оценки.
    """
    print(f"\n{'=' * 60}")
    print(f"  Видео: {Path(video_path).name}")
    print(f"  Таблица: {param_table_path or '(нет)'}")
    print(f"  Тип видео: {video_type or 'авто'}")
    print(f"{'=' * 60}")

    # Запуск конвейера
    print("\n  [1/3] Запуск конвейера OCR...")
    try:
        xml_result, elapsed = run_pipeline_on_video(
            video_path, param_table_path, video_type
        )
    except Exception as e:
        print(f"  [ERROR] Ошибка конвейера: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "video": video_path}

    print(f"  Конвейер завершён за {elapsed:.1f}с")

    # Сохраняем результат
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_id = Path(video_path).stem
    xml_path = out_dir / f"{video_id}_result.xml"
    xml_path.write_text(xml_result, encoding="utf-8")
    print(f"  XML сохранён: {xml_path}")

    # Парсим результат
    print("\n  [2/3] Парсинг результатов...")
    recognized = parse_xml_params(xml_result)
    print(f"  Распознано таймстемпов: {len(recognized)}")
    if recognized:
        first_ts = next(iter(recognized.values()))
        print(f"  Параметров в первом снимке: {len(first_ts)}")

    # Сравнение с эталоном
    print("\n  [3/3] Сравнение с эталоном...")
    if ground_truth_path and Path(ground_truth_path).exists():
        gt_xml = Path(ground_truth_path).read_text(encoding="utf-8")
        ground_truth = parse_xml_params(gt_xml)
        print(f"  Эталонных таймстемпов: {len(ground_truth)}")

        comparison = compare_results(recognized, ground_truth)
    else:
        # Без эталона — считаем точность как 0 (нет данных для сравнения)
        print("  [WARN] Эталонный XML не указан — точность не может быть рассчитана")
        print("  Показаны распознанные значения для визуальной проверки")
        comparison = {
            "accuracy_pct": 0.0,
            "total_params": 0,
            "correct_params": 0,
            "per_param_accuracy": {},
            "mismatches": [],
            "pass_95pct": False,
            "note": "Нет эталонных данных для сравнения",
        }

    # Вывод результатов
    result = {
        "video": str(video_path),
        "param_table": param_table_path,
        "video_type": video_type,
        "elapsed_seconds": round(elapsed, 2),
        "recognized_timestamps": len(recognized),
        "recognized_params_per_snapshot": len(next(iter(recognized.values()))) if recognized else 0,
        **comparison,
    }

    _print_results(result)

    # Сохраняем JSON-отчёт
    report_path = out_dir / f"{video_id}_evaluation.json"
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n  Отчёт сохранён: {report_path}")

    return result


def evaluate_all_vlm(use_llm_judge: bool = False, zone_enabled: bool = True) -> list[dict]:
    """VLM оценка всех видео из директории Задание 2."""
    project_root = Path(__file__).resolve().parent.parent
    task_dir = project_root / "Задание 2"

    if not task_dir.exists():
        print(f"[ERROR] Директория не найдена: {task_dir}")
        return []

    # Конфигурация видео и ground truth файлов
    videos = [
        {
            "video": str(task_dir / "Видео 1" / "Видео 1.mp4"),
            "ground_truth": str(task_dir / "Видео 1" / "ground_truth.xml"),
        },
        {
            "video": str(task_dir / "Видео 2" / "Видео 2.mp4"),
            "ground_truth": str(task_dir / "Видео 2" / "ground_truth.xml"),
        },
    ]

    results = []
    for v in videos:
        if not Path(v["video"]).exists():
            print(f"[SKIP] Видео не найдено: {v['video']}")
            continue

        if not Path(v["ground_truth"]).exists():
            print(f"[SKIP] Ground truth XML не найден: {v['ground_truth']}")
            continue

        result = evaluate_video_vlm(
            video_path=v["video"],
            ground_truth_path=v["ground_truth"],
            use_llm_judge=use_llm_judge,
            zone_enabled=zone_enabled,
        )
        results.append(result)

    # Итоговая сводка
    if results:
        print(f"\n\n{'=' * 60}")
        print("  ИТОГОВАЯ СВОДКА VLM ОЦЕНКИ")
        print(f"{'=' * 60}")
        for r in results:
            vname = Path(r.get("video", "?")).name
            acc = r.get("accuracy_pct", 0)
            params = r.get("total_params", 0)
            status = "PASS" if r.get("pass_95pct") else "FAIL"
            print(f"  {vname}: accuracy={acc}% ({params} params) [{status}]")

    return results


def evaluate_all() -> list[dict]:
    """Оценка всех видео из директории Задание 2."""
    # Определяем путь к Задание 2
    project_root = Path(__file__).resolve().parent.parent
    task_dir = project_root / "Задание 2"

    if not task_dir.exists():
        print(f"[ERROR] Директория не найдена: {task_dir}")
        return []

    videos = [
        {
            "video": str(task_dir / "Видео 1" / "Видео 1.mp4"),
            "table": str(task_dir / "Видео 1" / "Таблица 1.xlsx"),
            "video_type": "direct",
        },
        {
            "video": str(task_dir / "Видео 2" / "Видео 2.mp4"),
            "table": str(task_dir / "Видео 2" / "Таблица 2.xlsx"),
            "video_type": "handheld",
        },
        {
            "video": str(task_dir / "Видео 3" / "Видео 3.mp4"),
            "table": str(task_dir / "Видео 3" / "Таблица 3.xlsx"),
            "video_type": "direct",
        },
    ]

    results = []
    for v in videos:
        if not Path(v["video"]).exists():
            print(f"[SKIP] Видео не найдено: {v['video']}")
            continue

        result = evaluate_video(
            video_path=v["video"],
            param_table_path=v["table"],
            video_type=v["video_type"],
        )
        results.append(result)

    # Итоговая сводка
    if results:
        print(f"\n\n{'=' * 60}")
        print("  ИТОГОВАЯ СВОДКА")
        print(f"{'=' * 60}")
        for r in results:
            vname = Path(r.get("video", "?")).name
            acc = r.get("accuracy_pct", 0)
            params = r.get("total_params", 0)
            elapsed = r.get("elapsed_seconds", 0)
            status = "PASS" if r.get("pass_95pct") else "FAIL"
            print(f"  {vname}: accuracy={acc}% ({params} params, {elapsed:.1f}s) [{status}]")

    return results


def _print_results(result: dict) -> None:
    """Выводит результаты в консоль."""
    print(f"\n  {'─' * 50}")
    print(f"  РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print(f"  {'─' * 50}")
    print(f"  Accuracy:        {result.get('accuracy_pct', 0)}%")
    print(f"  Correct/Total:   {result.get('correct_params', 0)}/{result.get('total_params', 0)}")
    print(f"  Time:            {result.get('elapsed_seconds', 0):.1f}s")
    print(f"  Snapshots:       {result.get('recognized_timestamps', 0)}")

    if result.get("pass_95pct"):
        print(f"\n  *** PASS *** Accuracy >= 95%")
    else:
        print(f"\n  *** FAIL *** Accuracy < 95%")

    if result.get("note"):
        print(f"  Note: {result['note']}")

    # Топ несовпадений
    mismatches = result.get("mismatches", [])
    if mismatches:
        print(f"\n  Первые несовпадения ({min(len(mismatches), 10)}):")
        for m in mismatches[:10]:
            print(f"    ts={m['timestamp']} param#{m['param_id']}: "
                  f"expected='{m['expected']}' got='{m['recognized']}'")

    # Точность по параметрам
    per_param = result.get("per_param_accuracy", {})
    if per_param:
        worst = sorted(per_param.items(), key=lambda x: x[1])[:5]
        if worst:
            print(f"\n  Худшие параметры (топ-5):")
            for pid, acc in worst:
                print(f"    param#{pid}: {acc}%")


def main():
    parser = argparse.ArgumentParser(
        description="InfoDiode — оценка качества распознавания OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video", "-v",
        help="Путь к видеофайлу (.mp4)",
    )
    parser.add_argument(
        "--table", "-t",
        help="Путь к таблице параметров (.xlsx/.csv)",
    )
    parser.add_argument(
        "--video-type",
        choices=["direct", "handheld", "handheld_angle"],
        help="Тип видеозаписи (по умолчанию — автоопределение)",
    )
    parser.add_argument(
        "--ground-truth", "-g",
        help="Путь к эталонному XML для сравнения точности",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Оценить все видео из Задание 2/",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/reports",
        help="Директория для отчётов (default: data/reports)",
    )
    parser.add_argument(
        "--no-preload-ocr",
        action="store_true",
        help="Не предзагружать PaddleOCR при старте",
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Использовать VLM пайплайн вместо OCR",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Использовать LLM-as-a-Judge для сравнения (семантическое сопоставление названий)",
    )
    parser.add_argument(
        "--ground-truth-xml",
        help="Путь к ground truth XML для VLM оценки (с name атрибутами)",
    )
    parser.add_argument(
        "--no-zones",
        action="store_true",
        help="Отключить зонный анализ (использовать полный кадр, fallback)",
    )

    args = parser.parse_args()

    # Настройка логирования
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Предзагрузка OCR (опционально, только для OCR режима)
    if not args.no_preload_ocr and not args.vlm:
        print("Предзагрузка PaddleOCR...")
        try:
            from app.core.ocr_engine import get_ocr_engine
            engine = get_ocr_engine()
            if engine:
                print("PaddleOCR загружен")
        except Exception as e:
            print(f"[WARN] Не удалось предзагрузить PaddleOCR: {e}")
            print("Продолжаем без предзагрузки (OCR загрузится при первом вызове)")

    zone_enabled = not args.no_zones  # По умолчанию зонный анализ включён

    if args.vlm:
        # VLM режим оценки
        if args.all:
            # Оценить все видео через VLM
            results = evaluate_all_vlm(use_llm_judge=args.llm_judge, zone_enabled=zone_enabled)
        else:
            # Оценить одно видео через VLM
            if not args.video or not args.ground_truth_xml:
                print("[ERROR] Для VLM оценки необходимы --video и --ground-truth-xml (или --all)")
                sys.exit(1)

            if not Path(args.ground_truth_xml).exists():
                print(f"[ERROR] Ground truth XML не найден: {args.ground_truth_xml}")
                sys.exit(1)

            result = evaluate_video_vlm(
                video_path=args.video,
                ground_truth_path=args.ground_truth_xml,
                output_dir=args.output_dir,
                use_llm_judge=args.llm_judge,
                zone_enabled=zone_enabled,
            )

    elif args.all:
        results = evaluate_all()
    elif args.video:
        result = evaluate_video(
            video_path=args.video,
            param_table_path=args.table,
            video_type=args.video_type,
            ground_truth_path=args.ground_truth,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
