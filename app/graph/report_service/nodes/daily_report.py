import json

from app.clients.vllm_client import invoke_qwen_hf
from app.graph.report_service.state import ReportState


async def daily_report_node(state: ReportState) -> dict:
    data = state.get("daily_data") or {}
    critic_issues = state.get("critic_issues") or []

    system_msg = "당신은 공장 관리자를 위한 배터리 수율 및 결함 종합 통계 분석가입니다."

    summary = data.get("summaryData", {})
    defects_json_str = json.dumps(summary.get("defects", []), ensure_ascii=False, indent=2)
    report_date = data.get("reportDate", "알 수 없음")

    user_msg = f"""
    [일일 통계 데이터]
    - 리포트 기준일: {report_date}
    - 총 검사 수 (totalCount): {summary.get("totalCount", 0)}
    - 양품 판정 수 (passCount): {summary.get("passCount", 0)}
    - 불량 판정 수 (rejectCount): {summary.get("rejectCount", 0)}
    - 분석 실패 수 (failedCount): {summary.get("failedCount", 0)}

    [전일 대비 비교 데이터]
    - 전일 총 검사 수 (prevTotalCount): {summary.get("prevTotalCount", 0)}
    - 전일 불량 판정 수 (prevRejectCount): {summary.get("prevRejectCount", 0)}

    [결함 유형별 발생 건수 (상세)]
    {defects_json_str}

    위 데이터를 바탕으로 공장장 및 생산 관리자가 하루의 공정 상태를 파악하고 내일의 생산 전략을 세울 수 있는 '일일 품질 종합 보고서'를 작성해줘.
    단, 아래 주의 사항을 지켜줘.

    [주의 사항]
    ** 웹 렌더링을 위해 전체 출력 형태는 반드시 위 마크다운 템플릿 형식을 엄격하게 지켜줘. **
    ** 증감률이나 수율, 비율 등을 계산할 때는 소수점 첫째 자리까지만 간단히 표기해. **
    ** 한국어로 작성해 줘. **
    ** 없는 정보를 억지로 생성해서 추론하지 말아줘**

    [보고서 출력 마크다운 템플릿]
    반드시 아래 제공된 마크다운 및 HTML 태그 템플릿 구조를 그대로 사용하여 작성할 것. 내용만 상황에 맞게 분석하여 채워줘.

    ### {report_date}의 생산 및 수율 요약
    | 항목 | 금일 실적 | 전일 대비 분석 |
    | :--- | :--- | :--- |
    | **총 검사 수** | {summary.get("totalCount", 0)} 건 | [전일 대비 검사량 증감 요약] |
    | **양품 (PASS)** | {summary.get("passCount", 0)} 건 | - |
    | **불량 (REJECT)** | {summary.get("rejectCount", 0)} 건 | [전일 대비 불량 건수 증감 요약] |
    | **분석 실패** | {summary.get("failedCount", 0)} 건 | - |
    | **최종 수율(%)** | [금일 수율 계산]% | [전일 대비 수율 상승/하락 평가] |

    ### 주요 결함 발생 현황
    * **1위:** [가장 많이 발생한 결함명] - [건수]건 ([전체 불량 중 차지하는 비율]% 추정)
    * **2위:** [두 번째로 많이 발생한 결함명] - [건수]건 ([전체 불량 중 차지하는 비율]% 추정)
    * ...(발생한 모든 결함에 대해 위 양식으로 반복)
    * **분석 코멘트:** [오늘 발생한 결함들의 주된 특징이나 편중에 대한 1~2줄 요약]

    ### 총평 및 개선 제안 (Actionable Insights)
    > **일일 품질 총평:** [오늘 전체적인 품질 수준이 양호한지, 위험한지 종합 평가]
    >
    > **내일 공정 조치 권장 사항:**
    > - [ ] [최다 발생 결함을 줄이기 위해 내일 아침 점검해야 할 설비/공정 제안 1]
    > - [ ] [전일 대비 악화된 지표에 대한 개선 가이드 2]
    """

    if critic_issues:
        issues_text = "\n".join(
            f"  - 기준 {i['criterion']}: {i['description']}" for i in critic_issues
        )
        user_msg += f"""
    [이전 보고서 검수 결과 - 아래 오류를 반드시 수정하여 재작성하세요]
{issues_text}
    """

    response_text = await invoke_qwen_hf(
        system_msg=system_msg,
        prompt_text=user_msg,
    )

    return {
        "title": f"{report_date} 총 요약 보고서",
        "generated_report": response_text,
    }
