from app.graph.image_quality_inspection_service.cv_inspection import check_physical_quality
from app.graph.image_quality_inspection_service.state import QualityState

# 1. CV 검사 노드 (cv_inspection_node)
def cv_inspection_node(state: QualityState):
    images = state.get("images", [])
    image_type = state.get("imageType", "")
    
    cv_results = []
    vlm_targets = []
    
    for img in images:
        fail_type, desc = check_physical_quality(img["imageUrl"], image_type)
        if fail_type:
            # CV에서 불합격한 경우 바로 결과에 추가
            cv_results.append({
                "imageId": img["imageId"], 
                "failType": fail_type, 
                "description": f"[OpenCV 검출] {desc}"
            })
        else:
            # CV를 통과한 이미지만 VLM 타겟으로 분류
            vlm_targets.append(img)
            
    # 누적할 결과(cv_results)와 다음 노드로 넘길 이미지(vlm_targets) 반환
    return {"inspection_result": cv_results, "vlm_target_images": vlm_targets}