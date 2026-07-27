from app.schemas.request import DailyReportRequest
from app.schemas.response import DailyReportResponse
from app.graph.report_service.graph_builder import report_llm_model


async def generate_daily_report(req: DailyReportRequest) -> DailyReportResponse:
    initial_state = {
        "daily_data": req.daily_data.model_dump(),
        "generated_report": "",
        "title": "",
        "retry_count": 0,
        "critic_verdict": None,
        "critic_issues": None,
    }

    result = await report_llm_model.ainvoke(initial_state)

    return DailyReportResponse(
        title=result["title"],
        generated_report=result["generated_report"],
        critic_verdict=result.get("critic_verdict"),
        retry_count=result.get("retry_count", 0),
        critic_issues=result.get("critic_issues"),
    )
