from langgraph.graph import END, START, StateGraph

from app.graph.individual_report_service.state import ReportState
from app.graph.individual_report_service.nodes.critic import critic_node
from app.graph.individual_report_service.nodes.individual_report_node import individual_report_node
from app.graph.individual_report_service.edges import route_after_critic


def build_individual_graph():
    workflow = StateGraph(ReportState)

    workflow.add_node("individual_node", individual_report_node)
    workflow.add_node("critic_node", critic_node)

    workflow.add_edge(START, "individual_node")
    workflow.add_edge("individual_node", "critic_node")
    workflow.add_conditional_edges(
        "critic_node",
        route_after_critic,
        {"individual_node": "individual_node", "END": END},
    )

    return workflow.compile()

individual_report_model = build_individual_graph()