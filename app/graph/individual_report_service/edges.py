from typing import Literal

from app.graph.individual_report_service.state import ReportState

def route_after_critic(state: ReportState) -> Literal["individual_node", "END"]:
    if state.get("critic_verdict") == "PASS":
        return "END"
    # FAIL일 경우 최대 1회 재시도
    if state.get("retry_count", 0) <= 1:
        return "individual_node"
    return "END"