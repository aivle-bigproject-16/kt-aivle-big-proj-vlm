from typing import TypedDict, List, Optional,Set

class DefectInfo(TypedDict):
    imageType: str
    defectType: List[str]

class IndividualData(TypedDict):
    cellSerialNo: str
    inspectionId: int
    totalImages: int
    cellSize: Set[float]
    pointGroups:List[Set[float]]
    defectInfo: List[DefectInfo]

class ReportState(TypedDict):
    individual_data: Optional[IndividualData]
    generated_report: str
    title: str
    severity: float                      
    retry_count: int
    critic_verdict: Optional[str]        # "PASS" | "FAIL"
    critic_issues: Optional[List[dict]]  # [{criterion, description}, ...]