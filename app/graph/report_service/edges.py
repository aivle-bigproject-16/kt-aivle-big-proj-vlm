from typing import Literal

from app.graph.report_service.state import ReportState


def route_after_critic(state: ReportState) -> Literal["daily_node", "END"]:
    if state.get("critic_verdict") == "PASS":
        return "END"
    # FAIL: retry_count는 critic_node에서 이미 +1됨. 1회 재시도 허용
    if state.get("retry_count", 0) <= 1:
        return "daily_node"
    return "END"
