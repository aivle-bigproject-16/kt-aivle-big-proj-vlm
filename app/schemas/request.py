from typing import List
from pydantic import BaseModel


class DefectCountSchema(BaseModel):
    defectType: str
    count: int


class SummaryMetricsSchema(BaseModel):
    totalCount: int
    passCount: int
    rejectCount: int
    failedCount: int
    prevTotalCount: int
    prevRejectCount: int
    defects: List[DefectCountSchema]


class DailyDataSchema(BaseModel):
    reportDate: str
    summaryData: SummaryMetricsSchema


class DailyReportRequest(BaseModel):
    daily_data: DailyDataSchema


# ── Mockup ────────────────────────────────────────────────
MOCK_DAILY_REPORT_REQUEST = DailyReportRequest(
    daily_data=DailyDataSchema(
        reportDate="2026-07-06",
        summaryData=SummaryMetricsSchema(
            totalCount=120,
            passCount=104,
            rejectCount=14,
            failedCount=2,
            prevTotalCount=12,
            prevRejectCount=4,
            defects=[
                DefectCountSchema(defectType="MICRO_DEFECT", count=5),
                DefectCountSchema(defectType="CRACK", count=2),
            ],
        ),
    )
)
