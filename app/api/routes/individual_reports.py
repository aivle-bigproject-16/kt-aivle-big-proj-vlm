import json
from time import perf_counter

from fastapi import APIRouter, Response

from app.core.inference_trace import finish_trace, start_trace
from app.core.performance_metrics import PERFORMANCE_METRICS
from app.schemas.request import IndividualReportRequest, MOCK_INDIVIDUAL_REPORT_REQUEST
from app.schemas.response import ReportResponse
from app.services import individual_report_service

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post(
    "/individual",
    response_model=ReportResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": MOCK_INDIVIDUAL_REPORT_REQUEST.model_dump()
                }
            }
        }
    },
)
async def generate_individual_report(
    req: IndividualReportRequest,
    response: Response,
) -> ReportResponse:
    trace_token = start_trace()
    started_at = perf_counter()
    try:
        report = await individual_report_service.generate_individual_report(req)
    except Exception:
        elapsed_ms = max(0, round((perf_counter() - started_at) * 1000))
        PERFORMANCE_METRICS.record(
            "individual_report",
            elapsed_ms,
            success=False,
        )
        finish_trace(trace_token)
        raise

    elapsed_ms = max(0, round((perf_counter() - started_at) * 1000))
    calls = finish_trace(trace_token)
    PERFORMANCE_METRICS.record(
        "individual_report",
        elapsed_ms,
        success=report.status == "COMPLETED",
    )
    response.headers["X-VLM-Timings"] = json.dumps(
        {
            "total_ms": elapsed_ms,
            "retry_count": max(
                0,
                sum(
                    call["operation"] == "individual_generate"
                    for call in calls
                )
                - 1,
            ),
            "calls": calls,
        },
        separators=(",", ":"),
    )
    return report
