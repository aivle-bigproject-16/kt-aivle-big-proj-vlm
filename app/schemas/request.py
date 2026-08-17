from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


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


class DefectInfoSchema(BaseModel):
    imageType: str
    defectType: List[str]


class IndividualReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cellSerialNo: str
    inspectionId: Optional[int]
    totalImages: int
    cellSize: Optional[List[float]]
    pointGroups: List[List[float]]
    # Contract questions are still open: ratio units/ranges (0-1 vs 0-100) and
    # the pointGroups coordinate system. Keep these numeric shapes permissive.
    ctVoidRatio: Optional[float]
    rgbDefectRate: Optional[float]
    defectInfo: List[DefectInfoSchema]
    sourceInspectionIds: List[int] = Field(default_factory=list)
    finalLabel: Optional[str] = None
    inspectionStatus: Optional[str] = None
    failureType: Optional[str] = None
    failureReason: Optional[str] = None


# ── Mockup ────────────────────────────────────────────────
MOCK_INDIVIDUAL_REPORT_REQUEST = IndividualReportRequest(
    cellSerialNo="CELL-A92B-2026",
    inspectionId=84210,
    totalImages=12,
    cellSize=None,
    pointGroups=[],
    ctVoidRatio=None,
    rgbDefectRate=None,
    defectInfo=[
        DefectInfoSchema(imageType="CT", defectType=["MICRO_DEFECT"]),
        DefectInfoSchema(imageType="RGB", defectType=["CRACK", "SPOT"]),
        DefectInfoSchema(imageType="RGB", defectType=["SPOT"]),
    ],
    sourceInspectionIds=[84210, 84211],
    finalLabel="REJECT",
    inspectionStatus="COMPLETED",
    failureType=None,
    failureReason=None,
)

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
