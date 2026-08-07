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
    # [STEP 2] VLM (Qwen) 정밀 검사 (1 이미지 당 개별 호출로 변경)
    # -------------------------------------------------------------
    
    for img in vlm_target_images:
        target_id = img["imageId"]
        image_url = img["imageUrl"]

        if image_type == "CT":
            system_msg = "당신은 배터리 내부 구조 CT 검사 영상의 무결성을 판독하는 AI 품질 엔지니어입니다. 반드시 지정된 JSON 형식으로만 응답하세요."

            user_msg = f"""
            제공된 이미지는 배터리 셀의 내부 CT 촬영 영상입니다.
            분석 대상 이미지를 분석하여, 가장 유력한 이미지 화질 불량 원인 하나만 판별하세요.
            주의: 배터리가 화면에 꽉 차게 찍힌 정상적인 상태를 '잘림'으로 오인하지 마십시오.

            [분석 대상 안내]
            분석 대상 ID: {target_id}
            
            [판별 기준: CT 촬영 실패 케이스]
            * ct_cell_alignment_failure: 배터리 본체의 주요 영역이 화면 밖으로 완전히 벗어나 검사가 불가능할 정도로 크게 잘려나간 상태.
            * ct_low_signal_noise: 입자성 Poisson noise가 증가하고, 대비와 미세 구조 식별력이 저하됨.
            * ct_acquisition_motion: 구조 경계의 방향성 흐림, 동일 구조가 이동 방향으로 이중으로 보이는 ghosting.
            * ct_insufficient_projection_sampling: 구조 주변 streak, 방향성 aliasing, 경계·세부 구조의 재구성 손실.
            * ct_beam_hardening_metal_streak: 고밀도 영역 주변 cupping·명암 왜곡과 방사형 밝고 어두운 streak.
            * ct_NONE: 위 결함이 없는 정상적인 CT 영상.

            [출력 JSON 형식]
            반드시 아래와 같은 형태의 단일 JSON 객체를 포함하는 배열을 출력하세요.
            [
              {{"imageId":"{target_id}", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}}
            ]
            """
        else:
            system_msg = "당신은 배터리 외관 표면 RGB 검사 영상의 무결성을 판독하는 AI 품질 엔지니어입니다. 반드시 지정된 JSON 형식으로만 응답하세요."

            user_msg = f"""
            제공된 이미지는 배터리 셀의 외부 표면을 촬영한 RGB 영상입니다.
            분석 대상 이미지를 분석하여, 가장 유력한 이미지 화질 불량 원인 하나만 판별하세요.
            주의: 배터리가 화면에 꽉 차게 찍힌 정상적인 상태를 '잘림'으로 오인하지 마십시오.

            [분석 대상 안내]
            분석 대상 ID: {target_id}

            [판별 기준: RGB 촬영 실패 케이스]
            * rgb_trigger_timing_failure: 배터리의 상단이나 하단 캡 부분이 화면 밖으로 아예 잘려나가서 전체 형태가 온전하지 않은 심각한 상태.
            * rgb_uneven_lighting: 한쪽은 밝고 다른 쪽은 어두운 불균일한 이미지.
            * rgb_reflection_glare: 흰 반사광이 표면 scratch 및 오염 부위를 덮음.
            * rgb_surface_dust: 표면에 작은 점, 얼룩, 입자 오염.
            * rgb_hair_contamination: 길고 얇은 검정 또는 갈색 곡선이 표면을 가림.
            * rgb_NONE: 위 결함이 없는 정상적인 RGB 영상.

            [출력 형식]
            반드시 아래와 같은 형태의 단일 JSON 객체를 포함하는 배열을 출력하세요.
            [
              {{"imageId":"{target_id}", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}}
            ]
            """

        # 해당 이미지만 단독으로 전송하여 추론 (주의력 분산 방지)
        response_text = await invoke_qwen_hf(system_msg, user_msg, [image_url])
        clean_response = re.sub(r'```json\n|```', '', response_text).strip()

        try:
            data_list = json.loads(clean_response)
            
            # 리스트로 반환된 경우 extend, 단일 딕셔너리일 경우 append 처리
            if isinstance(data_list, list):
                results.extend(data_list)
            else:
                results.append(data_list)
                
        except json.JSONDecodeError:
            print(f"[{target_id}] JSON 파싱 에러 발생:", clean_response)

    return {"inspection_result": results}