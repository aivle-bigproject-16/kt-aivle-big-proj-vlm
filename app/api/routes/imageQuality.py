from fastapi import APIRouter
from app.schemas.request import QualityState, MOCK_IMAGE_QUALITY
from app.schemas.response import ImageResponse
from app.services import image_quality_service as image

router = APIRouter(prefix="", tags=["qualityInspection"])


@router.post(
    "/qualityInspection",
    response_model=ImageResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": MOCK_IMAGE_QUALITY.model_dump()
                }
            }
        }
    },
)
async def generate_daily_report(req: QualityState) -> ImageResponse:
    return await image.generate_image_inspection(req)