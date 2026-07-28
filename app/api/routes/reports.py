from fastapi import APIRouter
from app.schemas.request import DailyReportRequest, MOCK_DAILY_REPORT_REQUEST
from app.schemas.response import ReportResponse
from app.services import daily_report_service as report_service

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post(
    "/daily",
    response_model=ReportResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": MOCK_DAILY_REPORT_REQUEST.model_dump()
                }
            }
        }
    },
)
async def generate_daily_report(req: DailyReportRequest) -> ReportResponse:
    return await report_service.generate_daily_report(req)
