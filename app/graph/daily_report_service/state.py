from typing import TypedDict, List, Optional


class DefectCount(TypedDict):
    defectType: str
    count: int


class SummaryMetrics(TypedDict):
    totalCount: int
    passCount: int
    rejectCount: int
    failedCount: int
    prevTotalCount: int
    prevRejectCount: int
    defects: List[DefectCount]


class DailyData(TypedDict):
    reportDate: str
    summaryData: SummaryMetrics


class ReportState(TypedDict):
    daily_data: Optional[DailyData]
    generated_report: str
    title: str
    retry_count: int
    critic_verdict: Optional[str]        # "PASS" | "FAIL"
    critic_issues: Optional[List[dict]]  # [{criterion, description}, ...]
