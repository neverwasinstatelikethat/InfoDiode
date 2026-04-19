"""Сборщик метрик оценки качества."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class MetricsCollector:
    """Собирает и агрегирует метрики конвейера обработки."""

    total_frames: int = 0
    frames_processed: int = 0
    total_params: int = 0
    correct_params: int = 0
    latency_ms: list[float] = field(default_factory=list)
    zone_accuracies: dict[str, list[bool]] = field(default_factory=dict)
    # Дополнительные метрики для мониторинга без ground truth
    non_zero_params: int = 0  # Количество ненулевых значений
    confidence_values: list[float] = field(default_factory=list)  # Confidence от OCR

    def record_frame(
        self,
        latency_ms: float,
        param_results: dict[int, tuple[str, str | None]],
        confidence: float | None = None,
    ) -> None:
        """Записывает результаты обработки одного кадра.

        Args:
            latency_ms: Время обработки кадра в мс.
            param_results: param_id -> (recognized_value, ground_truth_value_or_None).
            confidence: Средняя уверенность OCR для кадра (опционально).
        """
        self.frames_processed += 1
        self.total_frames += 1
        self.latency_ms.append(latency_ms)

        if confidence is not None:
            self.confidence_values.append(confidence)

        for param_id, (recognized, ground_truth) in param_results.items():
            self.total_params += 1
            if ground_truth is not None and recognized == ground_truth:
                self.correct_params += 1
            # Подсчёт ненулевых значений как показатель качества детекции
            try:
                val = float(recognized.replace(",", ".").replace(" ", "").strip())
                if val != 0:
                    self.non_zero_params += 1
            except (ValueError, AttributeError):
                # Не число — считаем как "что-то обнаружено"
                if recognized and recognized.strip():
                    self.non_zero_params += 1

    def get_accuracy(self) -> float:
        """Возвращает общую точность распознавания.

        Returns:
            Точность в процентах (0-100).
            Если ground_truth недоступен, возвращает detection_rate.
        """
        if self.total_params == 0:
            return 0.0
        # Если есть ground_truth данные — возвращаем реальную точность
        if self.correct_params > 0:
            return self.correct_params / self.total_params * 100.0
        # Иначе — возвращаем detection_rate (доля ненулевых значений)
        return self.get_detection_rate()

    def get_detection_rate(self) -> float:
        """Возвращает долю ненулевых/валидных значений.

        Returns:
            Detection rate в процентах (0-100).
        """
        if self.total_params == 0:
            return 0.0
        return self.non_zero_params / self.total_params * 100.0

    def get_avg_confidence(self) -> float:
        """Возвращает среднюю уверенность OCR.

        Returns:
            Средняя уверенность (0-1) или 0 если нет данных.
        """
        if not self.confidence_values:
            return 0.0
        return sum(self.confidence_values) / len(self.confidence_values)

    def get_avg_latency(self) -> float:
        """Возвращает среднюю задержку обработки кадра.

        Returns:
            Среднее время в мс.
        """
        if not self.latency_ms:
            return 0.0
        return sum(self.latency_ms) / len(self.latency_ms)

    def get_p95_latency(self) -> float:
        """Возвращает P95 задержку.

        Returns:
            P95 время в мс.
        """
        if not self.latency_ms:
            return 0.0
        sorted_lat = sorted(self.latency_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def get_summary(self) -> dict:
        """Возвращает сводку метрик.

        Returns:
            Словарь с метриками.
        """
        return {
            "accuracy_pct": round(self.get_accuracy(), 2),
            "detection_rate_pct": round(self.get_detection_rate(), 2),
            "avg_confidence": round(self.get_avg_confidence(), 4),
            "total_params": self.total_params,
            "correct_params": self.correct_params,
            "non_zero_params": self.non_zero_params,
            "frames_processed": self.frames_processed,
            "avg_latency_ms": round(self.get_avg_latency(), 1),
            "p95_latency_ms": round(self.get_p95_latency(), 1),
        }
