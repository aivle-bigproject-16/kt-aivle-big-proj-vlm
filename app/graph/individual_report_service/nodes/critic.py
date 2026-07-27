import json

from app.clients.vllm_client import invoke_qwen_hf
from app.graph.individual_report_service.state import ReportState

async def critic_node(state: ReportState) -> dict:
    data = state.get("individual_data") or {}
    generated_report = state.get("generated_report", "")
    retry_count = state.get("retry_count", 0)

    system_msg = "당신은 배터리 셀 개별 검사 리포트의 수치와 형식을 검수하는 심사관입니다. 반드시 JSON 형식으로만 응답하세요."

    input_data_json = json.dumps(data, ensure_ascii=False, indent=2)

    user_msg = f"""
아래 원본 데이터와 생성된 보고서를 비교하여 오류를 찾아내세요.

[원본 데이터]
{input_data_json}

[생성된 보고서]
{generated_report}

[검수 기준]
1. 셀 정보 일치: Cell Serial No와 Inspection ID가 원본과 일치하는가?
2. 수치 일치: 총 검사 이미지 수가 원본의 totalImages와 일치하는가?
3. 날조 금지: 원본 데이터 detectedDefects에 없는 결함 유형이 보고서에 등장하지 않는가?

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
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        result = json.loads(text[start:end])
        verdict = result.get("verdict", "PASS")
        issues = result.get("issues", [])
    except (json.JSONDecodeError, ValueError):
        verdict = "PASS"
        issues = []

    update = {"critic_verdict": verdict, "critic_issues": issues}

    if verdict == "FAIL":
        update["retry_count"] = retry_count + 1

    return update