import json

from app.clients.vllm_client import invoke_qwen_hf
from app.graph.daily_report_service.state import ReportState


def _check_section_completeness(report: str, report_date: str) -> list[dict]:
    issues = []
    headers = [
        f"### {report_date}의 생산 및 수율 요약",
        "### 주요 결함 발생 현황",
        "### 총평 및 개선 제안",
    ]
    for h in headers:
        if h not in report:
            issues.append({"criterion": 1, "description": f"섹션 헤더 누락: '{h}'"})
    return issues


def _check_numbers(report: str, summary: dict) -> list[dict]:
    issues = []
    for field in ("totalCount", "passCount", "rejectCount", "failedCount"):
        value = str(summary.get(field, 0))
        if value not in report:
            issues.append({"criterion": 2, "description": f"{field} 수치 불일치: 원본={value}"})
    return issues


def _check_yield(report: str, summary: dict) -> list[dict]:
    total = summary.get("totalCount", 0)
    if total == 0:
        return []
    expected = round(summary.get("passCount", 0) / total * 100, 1)
    if str(expected) not in report:
        return [{"criterion": 3, "description": f"수율 불일치: 정확한 값={expected}%"}]
    return []


def _check_defect_ranking(report: str, defects: list) -> list[dict]:
    if not defects:
        return []
    sorted_defects = sorted(defects, key=lambda x: x.get("count", 0), reverse=True)
    positions = {d["defectType"]: report.find(d["defectType"]) for d in sorted_defects}

    issues = []
    for d in sorted_defects:
        if positions[d["defectType"]] == -1:
            issues.append({"criterion": 4, "description": f"결함명 미등장: {d['defectType']}"})

    found = [
        (d["defectType"], positions[d["defectType"]])
        for d in sorted_defects
        if positions[d["defectType"]] != -1
    ]
    for i in range(len(found) - 1):
        if found[i][1] > found[i + 1][1]:
            issues.append({
                "criterion": 4,
                "description": f"결함 순위 오류: {found[i][0]}이 {found[i+1][0]}보다 나중에 등장",
            })
    return issues


async def _check_hallucination(report: str, data: dict) -> list[dict]:
    system_msg = "배터리 보고서 검수관입니다. 반드시 JSON 형식으로만 응답하세요."
    user_msg = f"""원본 데이터에 없는 수치·결함명·제조사명이 보고서에 등장하는지 확인하세요.

[원본 데이터]
{json.dumps(data, ensure_ascii=False)}

[보고서]
{report}

[출력 형식]
{{"verdict": "PASS" 또는 "FAIL", "issues": [{{"criterion": 5, "description": "원본값 vs 보고서값"}}]}}
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
    data = state.get("daily_data") or {}
    generated_report = state.get("generated_report", "")
    retry_count = state.get("retry_count", 0)

    summary = data.get("summaryData", {})
    report_date = data.get("reportDate", "알 수 없음")

    # 규칙 기반 검수 (기준 1~4)
    issues: list[dict] = []
    issues += _check_section_completeness(generated_report, report_date)
    issues += _check_numbers(generated_report, summary)
    issues += _check_yield(generated_report, summary)
    issues += _check_defect_ranking(generated_report, summary.get("defects", []))

    # 모델 기반 검수 (기준 5: 날조 금지)
    issues += await _check_hallucination(generated_report, data)

    verdict = "FAIL" if issues else "PASS"
    update = {"critic_verdict": verdict, "critic_issues": issues}
    if verdict == "FAIL":
        update["retry_count"] = retry_count + 1
    return update
