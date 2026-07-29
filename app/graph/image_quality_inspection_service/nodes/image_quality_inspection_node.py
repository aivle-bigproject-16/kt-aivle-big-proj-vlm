import json

from app.clients.vllm_client import invoke_qwen_hf
from app.graph.image_quality_inspection_service.state import QualityState

async def image_quality_inspection_node(state: QualityState):
    images = state.get("images", [])
    image_type = state.get("imageType","")
    
    image_urls = [img["imagUrl"] for img in images]
    image_info_str = "\n".join([f"이미지 {i+1} - ID: {img['imageId']}" for i, img in enumerate(images)])

    if image_type == "CT":
      system_msg = "당신은 배터리 내부 구조 CT 검사 영상의 무결성을 판독하는 AI 품질 엔지니어입니다. 반드시 지정된 JSON 형식으로만 응답하세요."
      
      user_msg = f"""
      제공된 이미지는 배터리 셀의 내부 CT 촬영 영상입니다. 
      영상을 분석하여 어떤 촬영 장비 및 설정 오류로 인한 화질 불량인지 판별하세요.

      [입력된 이미지 정보]
      {image_info_str}

      [판별 기준: CT 촬영 실패 케이스]
      * ct_positioning_failure: 배터리 일부 잘림, 필요한 내부 영역 미촬영.
      * ct_fixation_motion: 경계 번짐, 구조가 이중으로 보이는 모션 아티팩트.
      * ct_detector_calibration: 동심원 ring 또는 줄무늬, 불량 픽셀 band 발생.
      * ct_low_xray_output: 전체 어두움, 강한 noise, 긴 금속 streak.
      * ct_high_xray_output: 재질 간 명암차 감소, 결함 경계 흐림.
      * ct_low_resolution: 작은 이물질, 미세 기공, 얇은 분리막 손실 또는 합쳐짐.
      * ct_sparse_projection_aliasing: 물체 주위 방사형 줄무늬 및 aliasing.
      * ct_photon_starvation: 금속 주변 심한 검은 영역과 국소 streak.
      * NONE: 위 결함이 없는 정상적인 CT 영상

      [출력 JSON 형식]
      [
        {{"imageId":"분석한 이미지 ID", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}},
        {{"imageId":"분석한 이미지 ID", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}}
      ]
      """
    else:
      system_msg = "당신은 배터리 외관 표면 RGB 검사 영상의 무결성을 판독하는 AI 품질 엔지니어입니다. 반드시 지정된 JSON 형식으로만 응답하세요."
      
      user_msg = f"""
      제공된 이미지는 배터리 셀의 외부 표면을 촬영한 RGB 영상입니다. 
      영상을 분석하여 어떤 물리적 환경 및 설정 오류로 인한 화질 불량인지 판별하세요.

      [입력된 이미지 정보]
      {image_info_str}

      [판별 기준: RGB 촬영 실패 케이스]
      * rgb_trigger_timing_failure: 배터리 앞부분 또는 뒷부분이 잘린 이미지.
      * rgb_alignment_failure: 한쪽으로 치우치거나 회전된 배터리.
      * rgb_uneven_lighting: 한쪽은 밝고 다른 쪽은 어두운 불균일한 이미지.
      * rgb_reflection_glare: 흰 반사광이 표면 scratch 및 오염 부위를 덮음.
      * rgb_focus_failure: 경계 흐림, 작은 scratch 미식별.
      * rgb_underexposure: 전체적으로 어둡고 noise가 많은 이미지.
      * rgb_overexposure: 밝은 부분 포화, 결함 정보 소실.
      * rgb_surface_dust: 표면에 작은 점, 얼룩, 입자 오염.
      * rgb_hair_contamination: 길고 얇은 검정 또는 갈색 곡선이 표면을 가림.
      * NONE: 위 결함이 없는 정상적인 RGB 영상

      [출력 JSON 형식]
      [
        {{"imageId":"분석한 이미지 ID", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}},
        {{"imageId":"분석한 이미지 ID", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}}
      ]
      """

    response_text = await invoke_qwen_hf(system_msg, user_msg, image_urls)

    return {"inspection_result": response_text}