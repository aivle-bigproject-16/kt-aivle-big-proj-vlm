import operator
from typing import TypedDict, List, Optional,Annotated

class Images(TypedDict):
    imageId: str
    imageUrl: str

class QualityState(TypedDict):
    imageType: Optional[str]
    images: List[Images]
    
    inspection_result: Annotated[List[dict], operator.add]