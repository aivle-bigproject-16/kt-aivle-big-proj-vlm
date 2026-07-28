from fastapi import APIRouter
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
    return await individual_report_service.generate_individual_report(req)
