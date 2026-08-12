import pytest
from unittest.mock import AsyncMock, patch

from app.graph.daily_report_service.edges import route_after_critic
from app.graph.daily_report_service.nodes.daily_report import daily_report_node

# 1. 그래프 엣지(조건부 라우팅) 로직 테스트
def test_route_after_critic():
    # 시나리오 A: 검수 PASS 시 종료(END)로 이동하는가?
    state_pass = {"critic_verdict": "PASS", "retry_count": 0}
    assert route_after_critic(state_pass) == "END"
    
    # 시나리오 B: 검수 FAIL이고 재시도 1회 이하일 때 생성 노드로 되돌아가는가?
    state_retry = {"critic_verdict": "FAIL", "retry_count": 0}
    assert route_after_critic(state_retry) == "daily_node"
    
    # 시나리오 C: 검수 FAIL이지만 재시도 횟수를 초과(2회 이상)했을 때 종료(END)하는가?
    state_stop = {"critic_verdict": "FAIL", "retry_count": 2}
    assert route_after_critic(state_stop) == "END"


# 2. 그래프 노드(Node) 상태 업데이트 테스트
@pytest.mark.asyncio
@patch("app.graph.daily_report_service.nodes.daily_report.invoke_qwen_hf", new_callable=AsyncMock)
async def test_daily_report_node(mock_invoke):
    # VLM의 텍스트 생성 결과를 가짜(Mock)로 설정
    mock_invoke.return_value = "### AI가 작성한 모의 보고서 본문"
    
    # 노드에 주입할 입력 상태(State)
    input_state = {
        "daily_data": {
            "reportDate": "2026-08-14",
            "summaryData": {"totalCount": 100}
        },
        "critic_issues": [] # 검수 지적 사항 없음
    }
    
    # 노드 함수 직접 실행
    result = await daily_report_node(input_state)
    
    # 상태(State)가 정상적으로 업데이트 되어 반환되는지 검증
    assert "2026-08-14" in result["title"]
    assert result["generated_report"] == "### AI가 작성한 모의 보고서 본문"
    
    # VLM 호출 함수가 1회 정상적으로 트리거 되었는지 확인
    mock_invoke.assert_called_once()