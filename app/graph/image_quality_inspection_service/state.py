from typing import TypedDict, List, Optional

class QualityState(TypedDict):
    image_type: Optional[str]
    image_urls: List[str]
    inspection_result: Optional[dict[str]]