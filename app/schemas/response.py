from typing import Optional
from pydantic import BaseModel


class DailyReportResponse(BaseModel):
    status: str              # "COMPLETED" | "FAILED"
    title: Optional[str]
    content: Optional[str]
    failureReason: Optional[str]
