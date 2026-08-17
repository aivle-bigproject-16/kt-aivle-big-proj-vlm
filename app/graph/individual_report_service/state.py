from typing import TypedDict, List, Optional

class DefectInfo(TypedDict):
    imageType: str
    defectType: List[str]

class IndividualData(TypedDict):
    cellSerialNo: str
    inspectionId: Optional[int]
    totalImages: int
    cellSize: Optional[List[float]]
    pointGroups: List[List[float]]
    ctVoidRatio: Optional[float]
    rgbDefectRate: Optional[float]
    defectInfo: List[DefectInfo]
    sourceInspectionIds: List[int]
    finalLabel: Optional[str]
    inspectionStatus: Optional[str]
    failureType: Optional[str]
    failureReason: Optional[str]

class ReportState(TypedDict):
    individual_data: Optional[IndividualData]
    generated_report: str
    title: str                 
    retry_count: int
    critic_verdict: Optional[str]        # "PASS" | "FAIL"
    critic_issues: Optional[List[dict]]  # [{criterion, description}, ...]
