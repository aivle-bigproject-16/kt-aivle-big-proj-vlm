import json

from app.graph.individual_report_service.state import ReportState


def build_individual_report_prompt(
    data: dict, critic_issues: list[dict] | None = None
) -> tuple[str, str, str]:
    critic_issues = critic_issues or []
    system_msg = (
        "당신은 배터리 제조 현장의 선임 품질 엔지니어입니다. "
        "제공된 수치와 상태만 사용해 간결하고 자연스러운 한국어 검사 리포트를 작성합니다. "
        "근거가 없는 원인, 위치, 확률, 결함을 추정하지 않습니다."
    )

    serial_no = data.get("cellSerialNo", "알 수 없음")
    inspection_id = data.get("inspectionId")
    source_inspection_ids = data.get("sourceInspectionIds", [])
    total_images = data.get("totalImages", 0)
    final_label = data.get("finalLabel") or "UNKNOWN"
    inspection_status = data.get("inspectionStatus") or "UNKNOWN"
    failure_type = data.get("failureType")
    failure_reason = data.get("failureReason")
    cell_size = data.get("cellSize")
    point_groups = data.get("pointGroups", [])
    ct_void_ratio = data.get("ctVoidRatio")
    rgb_defect_rate = data.get("rgbDefectRate")
    defects = data.get("defectInfo", [])

    defect_list = [item for item in defects if item.get("defectType")]
    defect_images = len(defect_list)
    ct_defects = sum(
        len(item.get("defectType", []))
        for item in defect_list
        if item.get("imageType") == "CT"
    )
    rgb_defects = sum(
        len(item.get("defectType", []))
        for item in defect_list
        if item.get("imageType") == "RGB"
    )
    location_rule = (
        "점군 좌표를 근거로 분포와 축 방향을 요약"
        if point_groups
        else "위치 데이터가 없으므로 위치 특성을 판단할 수 없다고 명시"
    )
    failure_rule = (
        f"분석 실패 유형은 {failure_type}, 사유는 {failure_reason}. "
        "셀 품질의 PASS/REJECT를 추정하지 말고 재검사를 권고"
        if final_label == "FAIL"
        else "분석이 정상 완료되었음을 반영"
    )

    user_msg = f"""
[개별 검사 데이터]
- Cell Serial No: {serial_no}
- Representative Inspection ID: {inspection_id}
- Source Inspection IDs: {source_inspection_ids}
- 총 검사 이미지 수: {total_images}
- 최종 판정: {final_label}
- 검사 상태: {inspection_status}
- 실패 유형: {failure_type}
- 실패 사유: {failure_reason}

[결함 및 위치 데이터]
- 결함 발견 이미지 수: {defect_images}
- CT 결함 수: {ct_defects}
- RGB 결함 수: {rgb_defects}
- 셀 크기: {cell_size}
- CT 점군 좌표: {point_groups}
- CT 공극 비율: {ct_void_ratio}
- RGB 불량 면적 비율: {rgb_defect_rate}
- 상세 결함: {json.dumps(defects, ensure_ascii=False)}

[작성 규칙]
- 한국어 Markdown으로 작성합니다.
- 최종 판정은 반드시 {final_label}로 그대로 표기합니다.
- {location_rule}합니다.
- {failure_rule}합니다.
- 상세 결함에 없는 유형이나 수량을 만들지 않습니다.
- 값이 None이거나 빈 배열이면 측정값이 없다고 명시하고 추정하지 않습니다.

[출력 형식]
### 셀 기본 정보
* **Cell Serial No:** {serial_no}
* **대표 검사 ID:** {inspection_id}
* **연결 검사 ID:** {source_inspection_ids}
* **총 검사 이미지 수:** {total_images}장
* **최종 판정:** {final_label}
* **검사 상태:** {inspection_status}

### 결함의 위치적 특성
* **분석 코멘트:** [위치 데이터 근거만 1~2문장으로 작성]

### 검사 요약 및 이미지 타입별 현황
* **결함 이미지:** {total_images}장 중 {defect_images}장
* **CT 이미지:** 결함 {ct_defects}건
* **RGB 이미지:** 결함 {rgb_defects}건

### 주요 결함 유형 분석
* **결함 유형:** [실제 유형과 건수만 작성. 없으면 '탐지된 결함 없음']
* **분석 코멘트:** [PASS/REJECT이면 근거 기반 요약, FAIL이면 실패 사유와 재검사 권고]
"""

    if critic_issues:
        issues_text = "\n".join(
            f"- 기준 {item['criterion']}: {item['description']}"
            for item in critic_issues
        )
        user_msg += f"""

[이전 보고서 검수 오류]
{issues_text}
위 오류를 모두 수정해 같은 형식으로 다시 작성합니다.
"""

    return serial_no, system_msg, user_msg


async def individual_report_node(state: ReportState) -> dict:
    data = state.get("individual_data") or {}
    critic_issues = state.get("critic_issues") or []
    serial_no, system_msg, user_msg = build_individual_report_prompt(
        data, critic_issues
    )

    from app.clients.vllm_client import invoke_qwen_hf

    response_text = await invoke_qwen_hf(
        system_msg=system_msg,
        prompt_text=user_msg,
    )

    return {
        "title": f"Cell [{serial_no}] 개별 검사 리포트",
        "generated_report": response_text,
    }
