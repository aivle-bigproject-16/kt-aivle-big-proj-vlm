import json
import re

from app.graph.image_quality_inspection_service.cv_inspector import check_physical_quality
from app.clients.vllm_client import invoke_qwen_hf
from app.graph.image_quality_inspection_service.state import QualityState

async def image_quality_inspection_node(state: QualityState):
    images = state.get("images", [])
    image_type = state.get("imageType","")

    results = []
    vlm_target_images = []

    # -------------------------------------------------------------
    # [STEP 1] OpenCV 사전 물리적 결함 검사 (RGB & CT 공통 적용)
    # -------------------------------------------------------------
    for img in images:
        target_id = img["imageId"]
        image_url = img["imageUrl"]
        
        # image_type 인자를 넘겨 RGB와 CT에 맞는 개별 검사 수행
        fail_type, desc = check_physical_quality(image_source=image_url, image_type=image_type)
        
        if fail_type:
            # OpenCV에서 불량으로 확정된 경우 VLM 분석 생략
            results.append({
                "imageId": target_id,
                "failType": fail_type,
                "description": f"[OpenCV 자동검출] {desc}"
            })
        else:
            # 사전 검사(잘림, 초점, 노출, 노이즈 등)를 통과한 이미지만 VLM으로 인계
            vlm_target_images.append(img)

    # 모두 불량 판정되어 VLM에 넘길 이미지가 없다면 즉시 반환
    if not vlm_target_images:
        return {"inspection_result": results}

    # -------------------------------------------------------------
    # [STEP 2] VLM (Qwen) 정밀 검사
    # -------------------------------------------------------------
    all_image_urls = []
    context_parts = []

    for idx, img in enumerate(vlm_target_images):
        target_idx = idx + 1
        target_id = img["imageId"]

        context_parts.append(f"이미지 {target_idx}번의 분석 대상 ID: {target_id}")
        all_image_urls.append(img["imageUrl"])

    context = "\n".join(context_parts)

    if image_type == "CT":
        system_msg = "당신은 배터리 내부 구조 CT 검사 영상의 무결성을 판독하는 AI 품질 엔지니어입니다. 반드시 지정된 JSON 형식으로만 응답하세요."
        user_msg = f"""
        제공된 이미지는 배터리 셀의 내부 CT 촬영 영상입니다. 당신에게 두 그룹(과거 불량 사례, 분석 대상 이미지)의 이미지가 번갈아서 제공됩니다.
        분석 대상 이미지를 분석하여, 가장 유력한 이미지 화질 불량 원인 하나만 판별하세요.

        [분석 세트 안내]
        {context}
        
        [판별 기준: CT 촬영 실패 케이스]
        * ct_cell_alignment_failure: 배터리 일부가 촬영 영역 밖으로 벗어나 외곽과 내부 구조 일부가 잘림.
        * ct_low_signal_noise: 입자성 Poisson noise가 증가하고, 대비와 미세 구조 식별력이 저하됨.
        * ct_acquisition_motion: 구조 경계의 방향성 흐림, 동일 구조가 이동 방향으로 이중으로 보이는 ghosting.
        * ct_insufficient_projection_sampling: 구조 주변 streak, 방향성 aliasing, 경계·세부 구조의 재구성 손실.
        * ct_beam_hardening_metal_streak: 고밀도 영역 주변 cupping·명암 왜곡과 방사형 밝고 어두운 streak.
        * ct_NONE: 위 결함이 없는 정상적인 CT 영상.

        [출력 JSON 형식]
        반드시 아래와 같이 입력된 모든 이미지에 대한 결과를 포함하는 순수 JSON 배열만 출력하세요.
        **반드시 이미지별로 빠짐 없이 json이 있어야합니다.**
        [
          {{"imageId":"분석 대상 ID", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}},
          {{"imageId":"분석 대상 ID", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}}
        ]
        """
    else:
        system_msg = "당신은 배터리 외관 표면 RGB 검사 영상의 무결성을 판독하는 AI 품질 엔지니어입니다. 반드시 지정된 JSON 형식으로만 응답하세요."
     
        user_msg = f"""
        제공된 이미지는 배터리 셀의 외부 표면을 촬영한 RGB 영상입니다. 당신에게 두 그룹(과거 불량 사례, 분석 대상 이미지)의 이미지가 번갈아서 제공됩니다.
        분석 대상 이미지를 분석하여, 가장 유력한 이미지 화질 불량 원인 하나만 판별하세요.

        [분석 세트 안내]
        {context}

        [판별 기준: RGB 촬영 실패 케이스]
        *rgb_trigger_timing_failure: 배터리 앞부분 또는 뒷부분이 잘린 이미지.
        * rgb_uneven_lighting: 한쪽은 밝고 다른 쪽은 어두운 불균일한 이미지.
        * rgb_reflection_glare: 흰 반사광이 표면 scratch 및 오염 부위를 덮음.
        * rgb_surface_dust: 표면에 작은 점, 얼룩, 입자 오염.
        * rgb_hair_contamination: 길고 얇은 검정 또는 갈색 곡선이 표면을 가림.
        * rgb_NONE: 위 결함이 없는 정상적인 RGB 영상.

        [출력 형식]
        반드시 아래와 같이 입력된 모든 이미지에 대한 결과를 포함하는 순수 JSON 배열만 출력하세요. 이미지당 오직 하나의 json을 만드세요.
        **반드시 이미지별로 빠짐 없이 json이 있어야합니다.**
        [
          {{"imageId":"분석 대상 ID", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}},
          {{"imageId":"분석 대상 ID", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}}
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