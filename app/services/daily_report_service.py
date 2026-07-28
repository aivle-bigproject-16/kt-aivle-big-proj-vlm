from app.schemas.request import DailyReportRequest
from app.schemas.response import ReportResponse
from app.graph.daily_report_service.graph_builder import report_llm_model


async def generate_daily_report(req: DailyReportRequest) -> ReportResponse:
    initial_state = {
        "daily_data": req.daily_data.model_dump(),
        "generated_report": "",
        "title": "",
        "retry_count": 0,
        "critic_verdict": None,
        "critic_issues": None,
    }

    try:
        result = await report_llm_model.ainvoke(initial_state)
        return ReportResponse(
            status="COMPLETED",
            title=result["title"],
            content=result["generated_report"],
            failureReason=None,
        )
    except Exception as e:
        return ReportResponse(
            status="FAILED",
            title=None,
            content=None,
            failureReason=str(e),
        )
