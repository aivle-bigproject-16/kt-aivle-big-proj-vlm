from fastapi import APIRouter
from time import perf_counter

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
async def generate_individual_report(req: IndividualReportRequest) -> ReportResponse:
    started_at = perf_counter()
    try:
        response = await individual_report_service.generate_individual_report(req)
    except Exception:
        elapsed_ms = max(0, round((perf_counter() - started_at) * 1000))
        PERFORMANCE_METRICS.record(
            "individual_report",
            elapsed_ms,
            success=False,
        )
        raise

    elapsed_ms = max(0, round((perf_counter() - started_at) * 1000))
    PERFORMANCE_METRICS.record(
        "individual_report",
        elapsed_ms,
        success=response.status == "COMPLETED",
    )
    return response
