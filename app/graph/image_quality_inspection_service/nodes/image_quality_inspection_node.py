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
            image_url = [".app/data/ct_NONE.jpg", ".app/data/ct_insufficient_projection_sampling.jpg", ".app/data/ct_cell_alignment_failure.jpg", ".app/data/ct_beam_hardening_metal_streak.jpg", ".app/data/ct_acquisition_motion.jpg"] + [image_url] 

            system_msg = "당신은 배터리 내부 구조 CT 검사 영상의 무결성을 판독하는 AI 품질 엔지니어입니다. 반드시 지정된 JSON 형식으로만 응답하세요."

            user_msg = f"""
            제공된 3장의 이미지를 순서대로 비교 분석하세요.
            
            [이미지 순서 및 역할]
            * 1번째 이미지 (Reference): 완벽하게 깨끗한 정상(rgb_NONE) 배터리의 표본입니다.
            * 2번째 이미지 (Reference): 표면에 먼지/얼룩이 묻은 불량(ct_insufficient_projection_sampling)의 표본입니다.
            * 3번째 이미지 (Reference): 표면에 먼지/얼룩이 묻은 불량(ct_cell_alignment_failure)의 표본입니다.
            * 4번째 이미지 (Reference): 표면에 먼지/얼룩이 묻은 불량(ct_beam_hardening_metal_streak)의 표본입니다.
            * 5번째 이미지 (Reference): 표면에 먼지/얼룩이 묻은 불량(ct_acquisition_motion)의 표본입니다.

            * 6번째 이미지 (Target): 당신이 분석해야 할 대상 이미지입니다. (ID: {target_id})

            [지시사항]
            1, 2,3,4,5번째 레퍼런스 이미지와 6번째 타겟 이미지를 시각적으로 꼼꼼히 비교하십시오. 
            타겟 이미지가 어떤 결함에 해당하는지 판별하세요.

            [분석 대상 안내]
            분석 대상 ID: {target_id}
            
            [판별 기준: CT 촬영 실패 케이스 및 핵심 관찰 포인트]
            1. ct_NONE: 아래 결함이 전혀 없는 매끄럽고 선명한 정상 CT 영상
            2. ct_acquisition_motion: [키워드: 이중 윤곽선, 방향성 흐림] 배터리의 테두리나 내부 층상 구조가 특정 방향으로 흔들려 두 겹(Ghosting)으로 보임.
            3. ct_insufficient_projection_sampling: [키워드: 알리어싱, 재구성 손실] 외곽선이 계단처럼 깨져 보이며, 구조 주변으로 얕은 줄무늬(Streak)가 전체적으로 발생함.
            4. ct_beam_hardening_metal_streak: [키워드: Cupping 왜곡, 강한 방사형 선] 특정 고밀도 영역을 중심으로 명암이 둥글게 왜곡되며, 밝고 어두운 강렬한 선들이 뻗어나감.
            5. ct_cell_alignment_failure: [키워드: 인위적 직선 잘림] 배터리 본체가 이미지 경계 밖으로 벗어나 테두리가 날카로운 일직선으로 잘려나감. (화면에 꽉 찬 둥근 테두리는 정상임)

            [출력 JSON 형식]
            반드시 아래와 같은 형태의 단일 JSON 객체를 포함하는 배열을 출력하세요.
            [
              {{"imageId":"{target_id}" , "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}}
            ]
            """
        else:
            system_msg = "당신은 배터리 외관 표면 RGB 검사 영상의 무결성을 판독하는 AI 품질 엔지니어입니다. 반드시 지정된 JSON 형식으로만 응답하세요."

            user_msg = f"""
            제공된 이미지는 배터리 셀의 외부 표면을 촬영한 RGB 영상입니다.
            분석 대상 이미지를 분석하여, 가장 유력한 이미지 화질 불량 원인 하나만 판별하세요.
            이미지의 시각적 특징을 먼저 논리적으로 분석한 뒤, 가장 유력한 상태를 하나만 판별하세요.

            [분석 대상 안내]
            분석 대상 ID: {target_id}

            [판별 기준: RGB 촬영 실패 케이스 및 핵심 관찰 포인트]
            1. rgb_NONE: 아래 결함이 전혀 없는 매끄럽고 선명한 정상 RGB 영상
            2. rgb_trigger_timing_failure: [키워드: 프레임 잘림] 배터리의 상/하 캡이나 몸통이 화면 밖으로 크게 잘려나감.
            3. rgb_uneven_lighting: [키워드: 명암 그라데이션] 화면 좌우 혹은 위아래 한쪽은 과도하게 밝고 반대쪽은 캄캄한 심한 조명 불균형이 있음.
            4. rgb_reflection_glare: [키워드: 하얀 빛기둥, 번짐] 표면에 굵고 길다란 하얀색 정반사 빛기둥(Core)과 빛 번짐(Bloom)이 발생해 표면을 가림.
            5. rgb_surface_dust: [키워드: 둥근 반점 얼룩] 초점이 흐린 원형 또는 타원형의 짙은 반점, 먼지 그림자가 묻어 있음.
            6. rgb_hair_contamination: [키워드: 가늘고 굽은 선] 화면을 뱀처럼 가로지르는 가늘고 구부러진 실오라기 같은 선형 그림자가 존재함.

            [출력 형식]
            반드시 아래와 같은 형태의 단일 JSON 객체를 포함하는 배열을 출력하세요.
            [
              {{"imageId":"{target_id}","failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}}
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