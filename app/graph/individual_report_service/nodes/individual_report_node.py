import json

from app.clients.vllm_client import invoke_qwen_hf
from app.graph.individual_report_service.state import ReportState


async def individual_report_node(state: ReportState) -> dict:
    data = state.get("individual_data") or {}
    critic_issues = state.get("critic_issues") or []

    system_msg = "당신은 배터리 셀 품질 검사 데이터를 분석하는 종합 통계 분석가입니다."

    # 데이터 추출 및 사전 집계
    serial_no = data.get("cellSerialNo", "알 수 없음")
    inspection_id = data.get("inspectionId", 0)
    total_images = data.get("totalImages", 0)
    defects = data.get("defectInfo", [])

    #cellSize = data.get("cellSize",{})
    #pointGroups = data.get("pointGroups",[])
    #공극체전률(CT) + 표면 불량면적 결합률(RGB)
    #위 3개로 한, 두 줄 간단 정리(위치 유형 분석 등등)

    defect_list = [d for d in defects if d.get("defectType") is not None]
    defects_images = len(defect_list)

    ct_defects = sum(
        len(d.get("defectType", []))
        for d in defect_list
        if d.get("imageType", "") == "CT"
    )
    rgb_defects = sum(
        len(d.get("defectType", []))
        for d in defect_list
        if d.get("imageType", "") == "RGB"
    )

    defects_json_str = json.dumps(defects, ensure_ascii=False, indent=2)

    user_msg = f"""
    [개별 검사 데이터]
    - Cell Serial No: {serial_no}
    - Inspection ID: {inspection_id}
    - 총 검사 이미지 수: {total_images}
    - 결함 발견 이미지 수: {defects_images}

    [이미지 타입별 통계]
    - CT: 총 {ct_defects}건
    - RGB: 총 {rgb_defects}건

    [상세 결함 리스트]
    {defects_json_str}

    위 데이터를 바탕으로 개별 셀에 대한 '검사 결과 리포트'를 작성해줘.

    [주의 사항]
    ** 웹 렌더링을 위해 전체 출력 형태는 반드시 위 마크다운 템플릿 형식을 엄격하게 지켜줘. **
    ** 한국어로 작성해 줘. **
    ** 없는 정보를 억지로 생성해서 추론하지 말아줘**

    [보고서 출력 마크다운 템플릿]
    ### 셀 기본 정보
    * **Cell Serial No:** {serial_no}
    * **Inspection ID:** {inspection_id}
    * **총 검사 이미지 수:** {total_images}장

    ### 검사 요약 및 이미지 타입별 현황
    * **전체 결함:** {total_images}장 중 {defects_images}장 결함
    * **CT 이미지:** 결함 {ct_defects}건
    * **RGB 이미지:**   결함 {rgb_defects}건

    ### 주요 결함 유형 분석
    * **[결함타입명]:** [발생 건수]건
    * **[결함타입명]:** [발생 건수]건
    * **분석 코멘트:** [현재 셀의 불량 양상에 대한 짧은 요약.]
    """

    if critic_issues:
        issues_text = "\n".join(
            f"  - 기준 {i['criterion']}: {i['description']}" for i in critic_issues
        )
        user_msg += f"""
    [이전 보고서 검수 결과 - 아래 오류를 반드시 수정하여 재작성하세요]
{issues_text}
    """

    # invoke_qwen_hf 함수는 기존에 정의된 것을 그대로 사용한다고 가정합니다.
    response_text = await invoke_qwen_hf(
        system_msg=system_msg,
        prompt_text=user_msg,
    )

    return {
        "title": f"Cell [{serial_no}] 개별 검사 리포트",
        "generated_report": response_text,
    }
