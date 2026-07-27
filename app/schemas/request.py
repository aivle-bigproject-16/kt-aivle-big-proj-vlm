from typing import List
from pydantic import BaseModel


class DefectCountSchema(BaseModel):
    defectType: str
    count: int


class PurchaseCountSchema(BaseModel):
    purchase: str
    count: int


class SummaryMetricsSchema(BaseModel):
    totalCount: int
    passCount: int
    rejectCount: int
    failedCount: int
    prevTotalCount: int
    prevRejectCount: int
    defects: List[DefectCountSchema]
    purchases: List[PurchaseCountSchema]


class DailyDataSchema(BaseModel):
    reportDate: str
    summaryData: SummaryMetricsSchema


class DailyReportRequest(BaseModel):
    daily_data: DailyDataSchema


# ── Mockup ────────────────────────────────────────────────
MOCK_DAILY_REPORT_REQUEST = DailyReportRequest(
    daily_data=DailyDataSchema(
        reportDate="2026-07-27",
        summaryData=SummaryMetricsSchema(
            totalCount=15000,
            passCount=14750,
            rejectCount=240,
            failedCount=10,
            prevTotalCount=14500,
            prevRejectCount=150,
            defects=[
                DefectCountSchema(defectType="CRACK", count=110),
                DefectCountSchema(defectType="SCRATCH", count=80),
                DefectCountSchema(defectType="DENT", count=50),
            ],
            purchases=[
                PurchaseCountSchema(purchase="LG", count=20),
                PurchaseCountSchema(purchase="삼성", count=3),
            ],
        ),
    )
)
