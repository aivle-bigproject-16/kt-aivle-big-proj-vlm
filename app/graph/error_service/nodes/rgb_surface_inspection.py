import json

from app.clients.vllm_client import invoke_qwen_hf
from app.graph.error_service.state import ErrorState

async def rgb_surface_inspection(state: ErrorState):
    image_urls = state.get("image_urls", [])

    system_msg = "당신은 배터리 외관 결함을 분석하는 품질 엔지니어입니다. 반드시 JSON으로만 응답하세요."
    
    user_msg = f"""
    제공된 RGB 이미지의 표면 상태를 분석하세요. 정상적인 바코드/텍스트 음각인지, 
    아니면 찍힘(Dent)이나 스크래치(Scratch) 같은 데미지인지 판별해야 합니다.
    
    [출력 JSON 형식]
    {{
      "type": "DAMAGE" 또는 "ENGRAVING"
    }}
    """
    
    response_text = await invoke_qwen_hf(system_msg, user_msg, image_urls)
    
    return {"inspection_result": response_text}