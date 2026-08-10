import json
import re
import asyncio

from app.graph.image_quality_inspection_service.cv_inspection import check_physical_quality
from app.clients.vllm_client import invoke_qwen_hf
from app.graph.image_quality_inspection_service.state import QualityState

# ==========================================
# [추가됨] 개별 이미지를 VLM으로 추론하는 비동기 독립 함수
# ==========================================
async def process_single_vlm(img: dict, image_type: str) -> list:
    target_id = img["imageId"]
    image_url = img["imageUrl"]

    if image_type == "CT":
        system_msg = "당신은 배터리 내부 구조 CT 검사 영상의 무결성을 판독하는 AI 품질 엔지니어입니다. 반드시 지정된 JSON 형식으로만 응답하세요."
        user_msg = f"""
        제공된 이미지는 배터리 셀의 내부 CT 촬영 영상입니다.
        분석 대상 이미지를 분석하여, 가장 유력한 이미지 화질 불량 원 하나만 판별하세요.

        [분석 대상 안내]
        분석 대상 ID: {target_id}
        
        [판별 기준: CT 촬영 실패 케이스]
        1. ct_NONE: 아래 결함이 전혀 없는 매끄럽고 선명한 정상 CT 영상
        2. ct_acquisition_motion: [키워드: 이중 윤곽선, 방향성 흐림] 배터리의 테두리나 내부 층상 구조가 특정 방향으로 흔들려 두 겹(Ghosting)으로 보임.
        3. ct_insufficient_projection_sampling: [키워드: 알리어싱, 재구성 손실] 외곽선이 계단처럼 깨져 보이며, 구조 주변으로 얕은 줄무늬(Streak)가 전체적으로 발생함.
        4. ct_beam_hardening_metal_streak: [키워드: Cupping 왜곡, 강한 방사형 선] 특정 고밀도 영역을 중심으로 명암이 둥글게 왜곡되며, 밝고 어두운 강렬한 선들이 뻗어나감.

        [출력 JSON 형식]
        반드시 아래와 같은 형태의 단일 JSON 객체를 포함하는 배열을 출력하세요.
        [
          {{"imageId":"{target_id}", "failType":"판별 기준에 명시된 실패 케이스 ID", "description":"발견된 현상에 대한 시각적 근거 요약"}}
        ]
        """

    elif image_type == "RGB":
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
    else:
        return []

    # 해당 이미지만 단독으로 전송하여 추론
    response_text = await invoke_qwen_hf(system_msg, user_msg, [image_url])
    clean_response = re.sub(r'```json\n|```', '', response_text).strip()

    try:
        data_list = json.loads(clean_response)
        # 통일성을 위해 무조건 리스트 형태로 감싸서 반환
        if isinstance(data_list, list):
            return data_list
        else:
            return [data_list]
            
    except json.JSONDecodeError:
        print(f"[{target_id}] JSON 파싱 에러 발생:", clean_response)
        # 에러 발생 시 프로그램이 멈추지 않도록 빈 리스트 반환
        return [] 


# ==========================================
# [수정됨] 메인 VLM 노드 (병렬 처리 적용)
# ==========================================
async def image_quality_inspection_node(state: QualityState):
    vlm_target_images = state.get("vlm_target_images", [])
    image_type = state.get("imageType", "")
    
    # 1. 처리해야 할 이미지들에 대한 비동기 작업(Task) 리스트 생성
    # (주의: 이 단계에서는 실행되지 않고 준비만 합니다)
    tasks = [
        process_single_vlm(img, image_type) 
        for img in vlm_target_images
    ]
    
    # 2. 모든 요청을 동시에(병렬로) 발사하고 모두 끝날 때까지 대기
    # gathered_results 형태: [[{결과1}], [{결과2}], ...]
    gathered_results = await asyncio.gather(*tasks)
    
    # 3. 개별 태스크가 반환한 이중 리스트들을 하나의 1차원 리스트로 병합
    vlm_results = []
    for res_list in gathered_results:
        vlm_results.extend(res_list)

    return {"inspection_result": vlm_results}