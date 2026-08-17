import json

from app.clients.vllm_client import invoke_qwen_hf
from app.graph.individual_report_service.report_quality import (
    check_failure_reinspection,
    check_korean_only,
)
from app.graph.individual_report_service.state import ReportState


def _check_cell_info(report: str, data: dict) -> list[dict]:
    issues = []
    if data.get("cellSerialNo", "") not in report:
        issues.append({"criterion": 1, "description": f"cellSerialNo 불일치: 원본={data.get('cellSerialNo')}"})
    if str(data.get("inspectionId", "")) not in report:
        issues.append({"criterion": 1, "description": f"inspectionId 불일치: 원본={data.get('inspectionId')}"})
    return issues


def _check_total_images(report: str, data: dict) -> list[dict]:
    value = str(data.get("totalImages", 0))
    if value not in report:
        return [{"criterion": 2, "description": f"totalImages 불일치: 원본={value}"}]
    return []


def _check_final_label(report: str, data: dict) -> list[dict]:
    value = data.get("finalLabel")
    if value and value not in report:
        return [{"criterion": 4, "description": f"finalLabel 누락: 원본={value}"}]
    return []


def _check_empty_defects(report: str, data: dict) -> list[dict]:
    defect_count = sum(
        len(item.get("defectType") or [])
        for item in data.get("defectInfo", [])
    )
    defect_tokens = ("MICRO_DEFECT", "CRACK", "SPOT", "SWELLING")
    if defect_count == 0 and any(token in report for token in defect_tokens):
        return [{"criterion": 5, "description": "결함 0건인데 결함 유형이 생성됨"}]
    return []


async def _check_hallucination(report: str, data: dict) -> list[dict]:
    system_msg = "배터리 보고서 검수관입니다. 반드시 JSON 형식으로만 응답하세요."
    user_msg = f"""원본 데이터에 없는 결함 유형이 보고서에 등장하는지 확인하세요.

[원본 데이터]
{json.dumps(data, ensure_ascii=False)}

[보고서]
{report}

[출력 형식]
{{"verdict": "PASS" 또는 "FAIL", "issues": [{{"criterion": 3, "description": "원본값 vs 보고서값"}}]}}
verdict가 PASS이면 issues는 빈 배열.
"""
    response = await invoke_qwen_hf(system_msg=system_msg, prompt_text=user_msg)
    try:
        text = response.strip()
        result = json.loads(text[text.find("{"):text.rfind("}") + 1])
        return result.get("issues", [])
    except (json.JSONDecodeError, ValueError):
        return []


async def critic_node(state: ReportState) -> dict:
    data = state.get("individual_data") or {}
    generated_report = state.get("generated_report", "")
    retry_count = state.get("retry_count", 0)

    # 규칙 기반 검수 (기준 1~2)
    issues: list[dict] = []
    issues += _check_cell_info(generated_report, data)
    issues += _check_total_images(generated_report, data)
    issues += _check_final_label(generated_report, data)
    issues += _check_empty_defects(generated_report, data)
    issues += check_korean_only(generated_report)
    issues += check_failure_reinspection(generated_report, data)

    # 모델 기반 검수 (기준 3: 날조 금지)
    issues += await _check_hallucination(generated_report, data)

    verdict = "FAIL" if issues else "PASS"
    update = {"critic_verdict": verdict, "critic_issues": issues}
    if verdict == "FAIL":
        update["retry_count"] = retry_count + 1
    return update
