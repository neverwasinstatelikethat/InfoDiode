"""Тесты для конвейера оценки качества."""

from app.evaluation.metrics_collector import MetricsCollector
from app.evaluation.report_generator import ReportGenerator


def test_metrics_accuracy() -> None:
    collector = MetricsCollector()
    collector.record_frame(100.0, {1: ("12.5", "12.5"), 2: ("3.14", "3.14")})
    collector.record_frame(150.0, {1: ("12.6", "12.5"), 2: ("3.14", "3.14")})
    # 3 correct out of 4 = 75%
    assert collector.get_accuracy() == 75.0


def test_metrics_latency() -> None:
    collector = MetricsCollector()
    collector.record_frame(100.0, {1: ("a", "a")})
    collector.record_frame(200.0, {1: ("b", "b")})
    collector.record_frame(300.0, {1: ("c", "c")})
    assert collector.get_avg_latency() == 200.0


def test_metrics_p95() -> None:
    collector = MetricsCollector()
    for i in range(100):
        collector.record_frame(float(i), {1: ("x", "x")})
    p95 = collector.get_p95_latency()
    assert 90.0 <= p95 <= 100.0


def test_metrics_summary() -> None:
    collector = MetricsCollector()
    collector.record_frame(50.0, {1: ("ok", "ok")})
    summary = collector.get_summary()
    assert summary["accuracy_pct"] == 100.0
    assert summary["total_params"] == 1
    assert summary["correct_params"] == 1
    assert summary["frames_processed"] == 1


def test_report_generator_json(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    gen = ReportGenerator(output_dir=str(tmp_path / "reports"))
    collector = MetricsCollector()
    collector.record_frame(100.0, {1: ("ok", "ok")})
    path = gen.generate_json_report(collector, "test_video", "direct")
    assert path.exists()
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["video_id"] == "test_video"
    assert data["pass_criteria"]["accuracy_95pct"] is True


def test_report_generator_markdown(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    gen = ReportGenerator(output_dir=str(tmp_path / "reports"))
    collector = MetricsCollector()
    collector.record_frame(100.0, {1: ("ok", "ok")})
    path = gen.generate_markdown_report(collector, "test_video", "direct")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "InfoDiode Evaluation Report" in content
    assert "PASS" in content


def test_metrics_empty() -> None:
    collector = MetricsCollector()
    assert collector.get_accuracy() == 0.0
    assert collector.get_avg_latency() == 0.0
    assert collector.get_p95_latency() == 0.0
    summary = collector.get_summary()
    assert summary["accuracy_pct"] == 0.0
