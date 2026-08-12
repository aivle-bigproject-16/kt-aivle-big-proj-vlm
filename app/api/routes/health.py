from fastapi import APIRouter, HTTPException, status
from app.clients.vllm_client import is_model_loaded

router = APIRouter(tags=["health"])


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently loading or unavailable."
        )
    
    return {
        "status": "ok", 
        "message": "Model is successfully loaded and ready to serve requests."
    }