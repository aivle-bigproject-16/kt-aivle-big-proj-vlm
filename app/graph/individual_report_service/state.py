from typing import TypedDict, List, Optional

class DefectInfo(TypedDict):
    imageType: str
    defectType: List[str]

class IndividualData(TypedDict):
    cellSerialNo: str
    inspectionId: int
    totalImages: int
    defectInfo: List[DefectInfo]

class ReportState(TypedDict):
    individual_data: Optional[IndividualData]
    generated_report: str
    title: str
    retry_count: int
    critic_verdict: Optional[str]        # "PASS" | "FAIL"
    critic_issues: Optional[List[dict]]  # [{criterion, description}, ...]