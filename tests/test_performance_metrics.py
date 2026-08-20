import asyncio
import json

from fastapi import Response

from app.api.routes import reports
from app.core.inference_trace import record_generation
from app.core.performance_metrics import PerformanceMetrics
from app.schemas.request import MOCK_DAILY_REPORT_REQUEST
from app.schemas.response import ReportResponse


def test_window_and_failure_counts_are_kept_separately():
    metrics = PerformanceMetrics(("daily_report",), window_size=3)

    for elapsed_ms in (10, 20, 30, 40):
        metrics.record("daily_report", elapsed_ms, success=True)
    metrics.record("daily_report", 50, success=False)

    daily = metrics.snapshot()["operations"]["daily_report"]
    assert daily["total_requests"] == 5
    assert daily["total_successes"] == 4
    assert daily["total_failures"] == 1
    assert daily["window_samples"] == 3
    assert daily["avg_ms"] == 30.0
    assert daily["p50_ms"] == 30
    assert daily["p95_ms"] == 40


def test_daily_report_route_records_end_to_end_latency(monkeypatch):
    metrics = PerformanceMetrics(("daily_report", "individual_report"))
    ticks = iter([100.0, 100.125])

    async def fake_generate_daily_report(req):
        record_generation(
            {
                "operation": "daily_generate",
                "generate_ms": 100,
                "input_tokens": 50,
                "output_tokens": 25,
            }
        )
        return ReportResponse(
            status="COMPLETED",
            title="title",
            content="content",
            failureReason=None,
        )

    monkeypatch.setattr(reports, "PERFORMANCE_METRICS", metrics)
    monkeypatch.setattr(reports, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        reports.report_service,
        "generate_daily_report",
        fake_generate_daily_report,
    )

    http_response = Response()
    report = asyncio.run(
        reports.generate_daily_report(
            MOCK_DAILY_REPORT_REQUEST,
            http_response,
        )
    )

    assert report.status == "COMPLETED"
    timing = json.loads(http_response.headers["X-VLM-Timings"])
    assert timing == {
        "total_ms": 125,
        "retry_count": 0,
        "calls": [
            {
                "operation": "daily_generate",
                "generate_ms": 100,
                "input_tokens": 50,
                "output_tokens": 25,
            }
        ],
    }
    daily = metrics.snapshot()["operations"]["daily_report"]
    assert daily["total_requests"] == 1
    assert daily["avg_ms"] == 125.0
