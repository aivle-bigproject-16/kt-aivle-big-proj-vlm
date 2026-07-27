from typing import Literal
from app.graph.error_service.state import ErrorState

def route_for_inspection_type(state: ErrorState) -> Literal["fail_image_quality_node", "rgb_surface_inspection"]:
    if state.get("inspection_type") == "RGB":
      return "rgb_surface_inspection"
    else:
      return "fail_image_quality_node"