from typing import TypedDict, List, Optional

class Images(TypedDict):
    imageId: str
    imageUrl: str

class QualityState(TypedDict):
    imageType: Optional[str]
    images: List[Images]

    reference_cases: Optional[List[dict[str, str]]]
    inspection_result: Optional[dict]