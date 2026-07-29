from langgraph.graph import END, START, StateGraph

from app.graph.image_quality_inspection_service.state import QualityState
from app.graph.image_quality_inspection_service.nodes.image_quality_inspection_node import image_quality_inspection_node
from app.graph.image_quality_inspection_service.nodes.reference_by_db import reference_by_db_node

def image_quality_inspection_graph():
    workflow = StateGraph(QualityState)
    
    # 노드 등록
    workflow.add_node("image_quality_inspection_node", image_quality_inspection_node)
    workflow.add_node("reference_by_db", reference_by_db_node)
    
    # 엣지 연결 (순차 실행: QA 검증 -> 외관 점검)
    workflow.add_edge(START, reference_by_db_node)
    workflow.add_edge(reference_by_db_node, image_quality_inspection_node)
    workflow.add_edge(image_quality_inspection_node, END)
    
    return workflow.compile()

image_inspection_graph = image_quality_inspection_graph()