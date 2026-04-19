"""OCR Logger — детальное логирование результатов OCR для диагностики.

Для каждого видео создаётся JSON-файл с результатами каждого кадра:
- Все распознанные тексты (Paddle + Florence)
- Извлечённые пары label:value
- Confidence каждого текста и пары
- Время обработки каждой стадии
- Fallback-события и причины

Формат лога:
{
    "video_id": "...",
    "created_at": "...",
    "config": { ... },
    "frames": [
        {
            "frame_idx": 0,
            "timestamp": "00:00:00.500",
            "is_duplicate": false,
            "stages": {
                "preprocess_ms": 12.3,
                "paddle_ms": 45.6,
                "pair_extractor_ms": 3.2,
                "florence_ms": 230.1,
                "fusion_ms": 1.1,
                "confidence_ms": 0.3,
                "total_ms": 292.6
            },
            "paddle_texts": [
                {"text": "TI-101", "bbox": [x,y,w,h], "confidence": 0.95, "box_type": "label"}
            ],
            "florence_texts": [
                {"text": "Давление котла", "bbox": [x,y,w,h], "confidence": 0.92, "box_type": "label"}
            ],
            "pairs": [
                {"label": "TI-101", "value": "758.3", "pair_confidence": 0.87, "relation": "horizontal"}
            ],
            "raw_fields": {"TI-101": "758.3"},
            "source": "merged",
            "overall_confidence": 0.85,
            "fallback": null
        }
    ],
    "summary": {
        "total_frames": 100,
        "duplicate_frames_skipped": 30,
        "total_pairs_found": 250,
        "total_fallbacks": 5,
        "avg_processing_ms": 310.5,
        "source_breakdown": {"paddle": 20, "merged": 50, "paddle_fallback": 5}
    }
}
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.core.ocr_models import RecognitionResult, TextBox, TextPair

logger = logging.getLogger(__name__)


def _bbox_to_list(bbox) -> list[int]:
    """Конвертирует BBox в список [x, y, w, h]."""
    return [bbox.x, bbox.y, bbox.w, bbox.h]


def _textbox_to_dict(tb: TextBox) -> dict:
    """Конвертирует TextBox в словарь для лога."""
    return {
        "text": tb.text,
        "bbox": _bbox_to_list(tb.bbox),
        "confidence": round(tb.confidence, 4),
        "box_type": tb.box_type,
    }


def _pair_to_dict(pair: TextPair) -> dict:
    """Конвертирует TextPair в словарь для лога."""
    return {
        "label": pair.label.text,
        "value": pair.value.text,
        "label_confidence": round(pair.label.confidence, 4),
        "value_confidence": round(pair.value.confidence, 4),
        "pair_confidence": round(pair.pair_confidence, 4),
        "relation": pair.relation,
    }


class OcrLogger:
    """Логгер результатов OCR — записывает детальный JSON-лог для каждого видео.

    Использование:
        ocr_log = OcrLogger(video_id="abc123")
        ocr_log.start_frame(0, "00:00:00.500")
        ocr_log.log_stage("paddle", 45.6, paddle_boxes)
        ocr_log.log_stage("pair_extractor", 3.2)
        ocr_log.log_stage("florence", 230.1, florence_boxes)
        ocr_log.log_result(result, is_duplicate=False, fallback=None)
        ...
        ocr_log.save()
    """

    def __init__(self, video_id: str, config: dict | None = None) -> None:
        self._video_id = video_id
        self._config = config or {}
        self._frames: list[dict] = []
        self._current_frame: dict | None = None
        self._total_pairs = 0
        self._total_fallbacks = 0
        self._source_counts: dict[str, int] = {}
        self._total_processing_ms = 0.0
        self._duplicate_count = 0

    def start_frame(self, frame_idx: int, timestamp: str) -> None:
        """Начинает запись нового кадра."""
        self._current_frame = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "is_duplicate": False,
            "stages": {},
            "paddle_texts": [],
            "florence_texts": [],
            "pairs": [],
            "raw_fields": {},
            "source": "",
            "overall_confidence": 0.0,
            "fallback": None,
        }

    def log_stage(
        self,
        stage_name: str,
        duration_ms: float,
        texts: list[TextBox] | None = None,
    ) -> None:
        """Логирует стадию обработки кадра.

        Args:
            stage_name: Название стадии (preprocess, paddle, pair_extractor, florence, fusion, confidence).
            duration_ms: Время выполнения стадии в мс.
            texts: Список TextBox (для paddle/florence стадий).
        """
        if self._current_frame is None:
            return

        self._current_frame["stages"][stage_name] = round(duration_ms, 1)

        if texts is not None:
            key = f"{stage_name}_texts"
            self._current_frame[key] = [_textbox_to_dict(tb) for tb in texts]

    def log_pairs(self, pairs: list[TextPair], stage_name: str = "pair_extractor") -> None:
        """Логирует извлечённые пары label:value.

        Args:
            pairs: Список TextPair.
            stage_name: Название стадии, которая произвела пары.
        """
        if self._current_frame is None:
            return

        self._current_frame["pairs"] = [_pair_to_dict(p) for p in pairs]

    def log_result(
        self,
        result: RecognitionResult,
        is_duplicate: bool = False,
        fallback: str | None = None,
    ) -> None:
        """Логирует финальный результат обработки кадра.

        Args:
            result: RecognitionResult.
            is_duplicate: Был ли кадр пропущен как дубликат.
            fallback: Причина fallback (null если не было).
        """
        if self._current_frame is None:
            return

        self._current_frame["is_duplicate"] = is_duplicate
        self._current_frame["raw_fields"] = result.raw_fields
        self._current_frame["source"] = result.source
        self._current_frame["overall_confidence"] = round(result.confidence, 4)
        self._current_frame["fallback"] = fallback

        if is_duplicate:
            self._duplicate_count += 1
        else:
            self._total_processing_ms += result.processing_ms

        # Обновляем пары если есть в результате
        if result.pairs and not self._current_frame["pairs"]:
            self._current_frame["pairs"] = [_pair_to_dict(p) for p in result.pairs]

        # Статистика
        self._total_pairs += len(result.raw_fields)
        source = result.source
        self._source_counts[source] = self._source_counts.get(source, 0) + 1
        if fallback:
            self._total_fallbacks += 1

        # Сохраняем кадр
        self._frames.append(self._current_frame)
        self._current_frame = None

    def log_fallback(
        self,
        ocr_texts: list | None = None,
        reason: str = "enhanced_pipeline_no_pairs",
    ) -> None:
        """Логирует событие fallback — когда enhanced pipeline не нашёл пар.

        Args:
            ocr_texts: Список текстов из прямого PaddleOCR (OCRTextResult или TextBox).
            reason: Причина fallback.
        """
        if self._current_frame is None:
            return

        self._current_frame["fallback"] = reason
        if ocr_texts:
            fallback_texts = []
            for t in ocr_texts:
                if hasattr(t, "text"):
                    entry = {"text": t.text}
                    if hasattr(t, "confidence"):
                        entry["confidence"] = round(t.confidence, 4)
                    if hasattr(t, "bbox"):
                        if hasattr(t.bbox, "x1"):
                            # OCRTextResult — нормализованные координаты
                            entry["bbox_normalized"] = [
                                round(t.bbox.x1, 4),
                                round(t.bbox.y1, 4),
                                round(t.bbox.x2, 4),
                                round(t.bbox.y2, 4),
                            ]
                        elif hasattr(t.bbox, "x"):
                            # TextBox — пиксельные координаты
                            entry["bbox"] = _bbox_to_list(t.bbox)
                    fallback_texts.append(entry)
            self._current_frame["fallback_texts"] = fallback_texts

    def save(self, output_dir: str | Path | None = None) -> Path:
        """Сохраняет лог в JSON-файл.

        Args:
            output_dir: Директория для сохранения. Если None — используется settings.output_xml_dir / "ocr_logs".

        Returns:
            Путь к сохранённому файлу.
        """
        if output_dir is None:
            from app.config import settings
            output_dir = Path(settings.output_xml_dir) / "ocr_logs"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_frames = len(self._frames)
        avg_ms = self._total_processing_ms / max(total_frames - self._duplicate_count, 1)

        log_data = {
            "video_id": self._video_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self._config,
            "frames": self._frames,
            "summary": {
                "total_frames": total_frames,
                "duplicate_frames_skipped": self._duplicate_count,
                "total_pairs_found": self._total_pairs,
                "total_fallbacks": self._total_fallbacks,
                "avg_processing_ms": round(avg_ms, 1),
                "source_breakdown": self._source_counts,
            },
        }

        log_path = output_dir / f"{self._video_id}_ocr_log.json"
        log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("OCR лог сохранён: %s (%d кадров)", log_path, total_frames)
        return log_path

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    @property
    def total_pairs(self) -> int:
        return self._total_pairs
