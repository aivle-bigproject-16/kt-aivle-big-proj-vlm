from langgraph.graph import END, START, StateGraph

from app.graph.error_service.nodes.fail_image_quality import fail_image_quality_node
from app.graph.error_service.nodes.rgb_surface_inspection import rgb_surface_inspection
from app.graph.error_service.edges import route_for_inspection_type
from app.graph.error_service.state import ErrorState

def build_inspection_graph():
    workflow = StateGraph(ErrorState)
    
    # 노드 등록
    workflow.add_node("fail_image_quality_node", fail_image_quality_node)
    workflow.add_node("rgb_surface_inspection", rgb_surface_inspection)
    
    # 엣지 연결 (순차 실행: QA 검증 -> 외관 점검)
    workflow.add_conditional_edges(START, route_for_inspection_type, {'fail_image_quality_node': 'fail_image_quality_node','rgb_surface_inspection': 'rgb_surface_inspection'})
    workflow.add_edge("rgb_surface_inspection", END)
    workflow.add_edge("fail_image_quality_node", END)
    
    return workflow.compile()

inspection_graph = build_inspection_graph()