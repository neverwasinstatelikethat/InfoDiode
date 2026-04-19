"""Единый контракт данных для OCR-движка.

Все модули OCR говорят на одном языке через эти dataclass-ы.
BBox — в пиксельных координатах выровненного кадра.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class BBox:
    """Bounding box в координатах выровненного кадра (пиксели)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        """Центр bbox."""
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def area(self) -> int:
        """Площадь bbox."""
        return self.w * self.h

    @property
    def quad(self) -> np.ndarray:
        """Четыре угла: TL, TR, BR, BL."""
        return np.array(
            [
                [self.x, self.y],
                [self.x + self.w, self.y],
                [self.x + self.w, self.y + self.h],
                [self.x, self.y + self.h],
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_quad(cls, quad: np.ndarray) -> BBox:
        """Создаёт BBox из массива точек полигона (N, 2)."""
        xs, ys = quad[:, 0], quad[:, 1]
        x, y = int(xs.min()), int(ys.min())
        w = int(xs.max()) - x
        h = int(ys.max()) - y
        return cls(x, y, max(w, 1), max(h, 1))

    def to_normalized(self, frame_h: int, frame_w: int) -> tuple[float, float, float, float]:
        """Конвертирует в нормализованные координаты [0,1] для API."""
        return (
            self.x / max(frame_w, 1),
            self.y / max(frame_h, 1),
            (self.x + self.w) / max(frame_w, 1),
            (self.y + self.h) / max(frame_h, 1),
        )

    def iou(self, other: BBox) -> float:
        """Intersection over Union с другим BBox."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - inter
        return inter / max(union, 1)


BoxType = Literal["label", "value", "mixed", "unknown"]
ColorTag = Literal[
    "value_green", "value_blue", "value_red", "value_white",
    "label", "status", "unknown"
]
RelationType = Literal["horizontal", "vertical", "inline"]
SourceType = Literal["paddle", "florence", "merged"]


@dataclass
class TextBox:
    """Единица распознавания: один текстовый блок на кадре."""

    bbox: BBox
    text: str
    confidence: float  # 0.0–1.0
    source: SourceType
    char_confs: list[float] = field(default_factory=list)  # по символам
    box_type: BoxType = "unknown"  # заполняется TextClassifier
    color_tag: ColorTag = "unknown"  # цветовая метка для SCADA индикаторов


@dataclass
class TextPair:
    """Пара label:value, извлечённая LayoutAnalyzer."""

    label: TextBox
    value: TextBox
    relation: RelationType
    pair_confidence: float  # уверенность в правильности связи


@dataclass
class RecognitionResult:
    """Финальный результат одного среза (500 мс)."""

    raw_fields: dict[str, str]  # {"Давление котла": "12.3 МПа"}
    confidence: float
    source: SourceType
    pairs: list[TextPair] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    frame_idx: int = 0
    processing_ms: float = 0.0
