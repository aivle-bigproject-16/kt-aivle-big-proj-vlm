from app.schemas.request import QualityState
from app.schemas.response import ImageResponse
from app.graph.image_quality_inspection_service.graph_builder import image_inspection_graph


async def generate_image_inspection(req: QualityState) -> ImageResponse:
    initial_state = {
        "imageType":req.imageType,
        "images":req.model_dump()["images"],
        "vlm_target_images": []
    }

    try:
        result = await image_inspection_graph.ainvoke(initial_state)
        return ImageResponse(
            status="COMPLETED",
            content=result["inspection_result"],
            failureReason=None,
        )
    except Exception as e:
        return ImageResponse(
            status="FAILED",
            content=None,
            failureReason=str(e),
        )