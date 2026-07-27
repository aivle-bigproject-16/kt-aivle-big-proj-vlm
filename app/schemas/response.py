from typing import List, Optional
from pydantic import BaseModel


class CriticIssueSchema(BaseModel):
    criterion: int
    description: str


class DailyReportResponse(BaseModel):
    title: str
    generated_report: str
    critic_verdict: Optional[str]
    retry_count: int
    critic_issues: Optional[List[CriticIssueSchema]]
