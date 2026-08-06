from typing import TypedDict, List, Optional

class Images(TypedDict):
    imageId: str
    imageUrl: str

class QualityState(TypedDict):
    imageType: Optional[str]
    images: List[Images]
    
    inspection_result: Optional[List[dict]]