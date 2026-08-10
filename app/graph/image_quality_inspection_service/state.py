from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator

class Images(TypedDict):
    imageId: str
    imageUrl: str

class QualityState(TypedDict):
    imageType: Optional[str]
    images: List[Images]
    vlm_target_images: List[Images] 
    
    inspection_result: Annotated[List[Dict[str, Any]], operator.add]