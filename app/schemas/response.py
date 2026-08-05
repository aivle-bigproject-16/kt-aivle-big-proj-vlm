from typing import Optional
from pydantic import BaseModel


class ReportResponse(BaseModel):
    status: str              # "COMPLETED" | "FAILED"
    title: Optional[str]
    content: Optional[str]
    failureReason: Optional[str]

class ImageResponse(BaseModel):
    status: str
    content: Optional[str]
    failureReason: Optional[str]
