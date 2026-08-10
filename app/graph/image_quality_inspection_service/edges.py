from app.graph.image_quality_inspection_service.state import QualityState

def route_to_vlm(state: QualityState):
    # 만약 VLM으로 넘길 이미지가 1장이라도 있으면 vlm_node로 이동
    if len(state.get("vlm_target_images", [])) > 0:
        return "vlm_node"
    # 통과한 이미지가 없으면 바로 종료
    return "END"