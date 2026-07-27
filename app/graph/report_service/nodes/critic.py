import json

from app.clients.vllm_client import invoke_qwen_hf
from app.graph.report_service.state import ReportState


async def critic_node(state: ReportState) -> dict:
    data = state.get("daily_data") or {}
    generated_report = state.get("generated_report", "")
    retry_count = state.get("retry_count", 0)

    system_msg = "당신은 배터리 공정 일일 보고서의 품질을 검수하는 심사관입니다. 반드시 JSON 형식으로만 응답하세요."

    summary = data.get("summaryData", {})
    report_date = data.get("reportDate", "알 수 없음")
    input_data_json = json.dumps(data, ensure_ascii=False, indent=2)

    user_msg = f"""
아래 원본 데이터와 생성된 보고서를 비교하여 오류를 찾아내세요.

[원본 데이터]
{input_data_json}

[생성된 보고서]
{generated_report}

[검수 기준]
1. 섹션 완전성: 아래 4개 섹션 헤더가 모두 존재하는가?
   - ### {report_date}의 생산 및 수율 요약
   - ### 주요 결함 발생 현황
   - ### 제조사별 결함 발생 현황
   - ### 총평 및 개선 제안
2. 수치 일치: 보고서 표의 totalCount·passCount·rejectCount·failedCount가 원본과 동일한가?
   - 원본 totalCount: {summary.get("totalCount", 0)}
   - 원본 passCount: {summary.get("passCount", 0)}
   - 원본 rejectCount: {summary.get("rejectCount", 0)}
   - 원본 failedCount: {summary.get("failedCount", 0)}
3. 수율 계산: 최종 수율(%) = round(passCount / totalCount * 100, 1) 과 일치하는가?
4. 결함 순위: defects를 count 내림차순 정렬한 순서와 보고서 순위가 일치하는가?
5. 제조사 순위: purchases를 count 내림차순 정렬한 순서와 보고서 순위가 일치하는가?
6. 날조 금지: 원본 데이터에 없는 수치, 결함명, 제조사명이 보고서에 등장하지 않는가? (단, 계산된 비율(%)은 예외로 허용)
7. 비율 검증: 계산 비율(%)이 수식으로 올바르게 계산되었는가?

[출력 형식 - 반드시 JSON으로만 응답]
{{
  "verdict": "PASS" 또는 "FAIL",
  "issues": [
    {{
      "criterion": 위반 기준 번호,
      "description": "구체적으로 어떤 값이 잘못되었는지 (원본값 vs 보고서값)"
    }}
  ]
}}
verdict가 PASS이면 issues는 빈 배열.
"""

    response_text = await invoke_qwen_hf(
        system_msg=system_msg,
        prompt_text=user_msg,
    )

    try:
        # 응답에서 JSON 블록만 추출
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        result = json.loads(text[start:end])
        verdict = result.get("verdict", "PASS")
        issues = result.get("issues", [])
    except (json.JSONDecodeError, ValueError):
        # 파싱 실패 시 PASS 처리하여 무한 루프 방지
        verdict = "PASS"
        issues = []

    update = {"critic_verdict": verdict, "critic_issues": issues}

    if verdict == "FAIL":
        update["retry_count"] = retry_count + 1

    return update
