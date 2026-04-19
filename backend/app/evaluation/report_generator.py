"""Генератор отчётов оценки качества."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.evaluation.metrics_collector import MetricsCollector


class ReportGenerator:
    """Генерирует отчёты оценки качества конвейера."""

    def __init__(self, output_dir: str = "data/reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json_report(
        self,
        collector: MetricsCollector,
        video_id: str,
        video_type: str,
    ) -> Path:
        """Генерирует JSON-отчёт.

        Args:
            collector: Сборщик метрик.
            video_id: ID видео.
            video_type: Тип видео (direct/handheld/handheld_angle).

        Returns:
            Путь к файлу отчёта.
        """
        report = self._build_report(collector, video_id, video_type)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"report_{video_id}_{timestamp}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def generate_markdown_report(
        self,
        collector: MetricsCollector,
        video_id: str,
        video_type: str,
    ) -> Path:
        """Генерирует Markdown-отчёт.

        Args:
            collector: Сборщик метрик.
            video_id: ID видео.
            video_type: Тип видео.

        Returns:
            Путь к файлу отчёта.
        """
        summary = collector.get_summary()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# InfoDiode Evaluation Report",
            f"",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Video ID | `{video_id}` |",
            f"| Video Type | `{video_type}` |",
            f"| Timestamp | {timestamp} |",
            f"| Accuracy | {summary['accuracy_pct']}% |",
            f"| Correct Params | {summary['correct_params']}/{summary['total_params']} |",
            f"| Frames Processed | {summary['frames_processed']} |",
            f"| Avg Latency | {summary['avg_latency_ms']} ms |",
            f"| P95 Latency | {summary['p95_latency_ms']} ms |",
            f"",
            f"## Pass/Fail",
            f"",
        ]

        if summary["accuracy_pct"] >= 95.0:
            lines.append(f"**PASS** — Accuracy {summary['accuracy_pct']}% >= 95%")
        else:
            lines.append(f"**FAIL** — Accuracy {summary['accuracy_pct']}% < 95%")

        if summary["avg_latency_ms"] <= 2000.0:
            lines.append(f"**PASS** — Avg latency {summary['avg_latency_ms']}ms <= 2000ms")
        else:
            lines.append(f"**FAIL** — Avg latency {summary['avg_latency_ms']}ms > 2000ms")

        path = self.output_dir / f"report_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _build_report(
        self,
        collector: MetricsCollector,
        video_id: str,
        video_type: str,
    ) -> dict[str, Any]:
        """Строит словарь отчёта.

        Args:
            collector: Сборщик метрик.
            video_id: ID видео.
            video_type: Тип видео.

        Returns:
            Словарь с данными отчёта.
        """
        summary = collector.get_summary()
        return {
            "video_id": video_id,
            "video_type": video_type,
            "timestamp": datetime.now().isoformat(),
            "metrics": summary,
            "pass_criteria": {
                "accuracy_95pct": summary["accuracy_pct"] >= 95.0,
                "avg_latency_2s": summary["avg_latency_ms"] <= 2000.0,
            },
        }
