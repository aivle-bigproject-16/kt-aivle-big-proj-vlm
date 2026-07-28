from app.schemas.request import IndividualReportRequest
from app.schemas.response import ReportResponse
from app.graph.individual_report_service.graph_builder import individual_report_model


async def generate_individual_report(req: IndividualReportRequest) -> ReportResponse:
    initial_state = {
        "individual_data": req.model_dump(),
        "generated_report": "",
        "title": "",
        "retry_count": 0,
        "critic_verdict": None,
        "critic_issues": None,
    }

    try:
        result = await individual_report_model.ainvoke(initial_state)
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
