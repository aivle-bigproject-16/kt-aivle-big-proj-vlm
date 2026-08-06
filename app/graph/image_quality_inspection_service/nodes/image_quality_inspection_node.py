import json
import re

from app.graph.image_quality_inspection_service.cv_inspector import check_physical_quality # OpenCV 모듈 임포트
from app.clients.vllm_client import invoke_qwen_hf
from app.graph.image_quality_inspection_service.state import QualityState

async def image_quality_inspection_node(state: QualityState):
    images = state.get("images", [])
    image_type = state.get("imageType","")

    reference_cases = state.get("reference_cases", [])
    ref_dict = {ref["target_imageId"]: ref for ref in reference_cases}

    results = []
    vlm_target_images = []

    if image_type == "RGB":
        for img in images:
            target_id = img["imageId"]
            image_url = img["imageUrl"]
            
            fail_type, desc = check_physical_quality(image_url)
            
            if fail_type:
                ref = ref_dict.get(target_id, {})
                results.append({
                    "imageId": target_id,
                    "ref_failtype": ref.get("ref_failType", "NONE"),
                    "failType": fail_type,
                    "description": f"[OpenCV 자동검출] {desc}"
                })
            else:
                # 정상일 경우 VLM 정밀 검사 대상으로 분류
                vlm_target_images.append(img)
    else:
        # CT 영상은 OpenCV 검사 없이 모두 VLM으로 넘김
        vlm_target_images = images

    # 모든 이미지가 OpenCV에서 불량 처리되어 VLM에 넘길 이미지가 없다면 즉시 반환
    if not vlm_target_images:
        return {"inspection_result": results}

    # -------------------------------------------------------------
    # [STEP 2] VLM (Qwen) 정밀 검사
    # -------------------------------------------------------------
    all_image_urls = []
    context_parts = []

    for img in vlm_target_images:
        target_id = img["imageId"]
        ref = ref_dict.get(target_id) 
        context_parts.append(f"\n--- 분석 대상 ID: {target_id} ---")
        
        if ref:
            all_image_urls.append(ref["ref_imageUrl"])
            ref_idx = len(all_image_urls)
            
            all_image_urls.append(img["imageUrl"])
            target_idx = len(all_image_urls)
            
            context_parts.append(
                f"* 이미지 {ref_idx} (과거 불량 사례): 판정 유형 [{ref['ref_failType']}]\n"
                f"* 이미지 {target_idx} (실제 분석 대상): 이미지 {ref_idx}(과거 사례)를 우선 참고하되, 두 이미지의 결함 양상이 명확히 다르다고 판단되면 과거 사례를 무시하고 [판별 기준]에 따라 독립적으로 판독하세요."
            )
        else:
            all_image_urls.append(img["imageUrl"])
            target_idx = len(all_image_urls)
            context_parts.append(
                f"* 이미지 {target_idx} (실제 분석 대상): 과거 참고 사례가 없습니다. 독립적으로 판독하세요."
            )
    context = "\n".join(context_parts)

    if image_type == "CT":
        system_msg = "당신은 배터리 내부 구조 CT 검사 영상의 무결성을 판독하는 AI 품질 엔지니어입니다. 반드시 지정된 JSON 형식으로만 응답하세요."
        user_msg = f"""
        제공된 이미지는 배터리 셀의 내부 CT 촬영 영상입니다. 당신에게 두 그룹(과거 불량 사례, 분석 대상 이미지)의 이미지가 번갈아서 제공됩니다.
        분석 대상 이미지를 분석하여, 가장 유력한 이미지 화질 불량 원인을 판별하세요.

        [분석 세트 안내]
        {context}
        
        [판별 기준: CT 촬영 실패 케이스]
        * ct_acquisition_motion: 구조 경계의 방향성 흐림, 동일 구조가 이동 방향으로 이중으로 보이는 ghosting.
        * ct_insufficient_projection_sampling: 구조 주변 streak, 방향성 aliasing, 경계·세부 구조의 재구성 손실.
        * ct_beam_hardening_metal_streak: 고밀도 영역 주변 cupping·명암 왜곡과 방사형 밝고 어두운 streak.
        * ct_NONE: 위 결함이 없는 정상적인 CT 영상.

        [출력 JSON 형식]
        반드시 아래와 같이 입력된 모든 이미지에 대한 결과를 포함하는 순수 JSON 배열만 출력하세요.
        [
          {{"imageId":"분석 이미지 ID", "ref_failtype": "참고한 이미지 실패 케이스", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}},
          {{"imageId":"분석 이미지 ID", "ref_failtype": "참고한 이미지 실패 케이스", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}}
        ]
        """
    else:
        system_msg = "당신은 배터리 외관 표면 RGB 검사 영상의 무결성을 판독하는 AI 품질 엔지니어입니다. 반드시 지정된 JSON 형식으로만 응답하세요."
     
        user_msg = f"""
        제공된 이미지는 배터리 셀의 외부 표면을 촬영한 RGB 영상입니다. 당신에게 두 그룹(과거 불량 사례, 분석 대상 이미지)의 이미지가 번갈아서 제공됩니다.
        분석 대상 이미지를 분석하여, 가장 유력한 이미지 화질 불량 원인을 판별하세요.

        [분석 세트 안내]
        {context}

        [판별 기준: RGB 촬영 실패 케이스]
        * rgb_uneven_lighting: 한쪽은 밝고 다른 쪽은 어두운 불균일한 이미지.
        * rgb_reflection_glare: 흰 반사광이 표면 scratch 및 오염 부위를 덮음.
        * rgb_surface_dust: 표면에 작은 점, 얼룩, 입자 오염.
        * rgb_hair_contamination: 길고 얇은 검정 또는 갈색 곡선이 표면을 가림.
        * rgb_NONE: 위 결함이 없는 정상적인 RGB 영상.

        [출력 형식]
        반드시 아래와 같이 입력된 모든 이미지에 대한 결과를 포함하는 순수 JSON 배열만 출력하세요.
        [
          {{"imageId":"분석 이미지 ID", "ref_failtype": "참고한 이미지 실패 케이스", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}},
          {{"imageId":"분석 이미지 ID", "ref_failtype": "참고한 이미지 실패 케이스", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}}
        ]
        """

    response_text = await invoke_qwen_hf(system_msg, user_msg, all_image_urls)
    clean_response = re.sub(r'```json\n|```', '', response_text).strip()

    try:
        data_list = json.loads(clean_response)
        results.extend(data_list)
    except json.JSONDecodeError:
        print("JSON 파싱 에러 발생:", clean_response)

    return {"inspection_result": results}