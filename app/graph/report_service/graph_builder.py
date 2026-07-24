from langgraph.graph import END, START, StateGraph

from app.graph.report_service.edges import route_after_critic
from app.graph.report_service.nodes.critic import critic_node
from app.graph.report_service.nodes.daily_report import daily_report_node
from app.graph.report_service.state import ReportState


def build_graph():
    workflow = StateGraph(ReportState)

    workflow.add_node("daily_node", daily_report_node)
    workflow.add_node("critic_node", critic_node)

    workflow.add_edge(START, "daily_node")
    workflow.add_edge("daily_node", "critic_node")
    workflow.add_conditional_edges(
        "critic_node",
        route_after_critic,
        {"daily_node": "daily_node", "END": END},
    )

    return workflow.compile()


report_llm_model = build_graph()
