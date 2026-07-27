from typing import TypedDict, List, Optional

class ErrorState(TypedDict):
    inspection_type: str
    image_type: Optional[str]
    image_urls: List[str]
    inspection_result: Optional[dict[str]]