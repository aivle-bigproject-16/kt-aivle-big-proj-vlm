import pytest
from pydantic import ValidationError

from app.graph.individual_report_service.nodes.individual_report_node import (
    build_individual_report_prompt,
)
from app.schemas.request import IndividualReportRequest


REQUEST_DATA = {
    "cellSerialNo": "SERIAL-PROMPT-8",
    "inspectionId": 987654321,
    "totalImages": 37,
    "cellSize": [101.25, 202.5, 7.75],
    "pointGroups": [[303.75, 404.5], [505.25, 606.0]],
    "ctVoidRatio": 0.123456,
    "rgbDefectRate": 87.654321,
    "defectInfo": [
        {"imageType": "RGB-SENTINEL", "defectType": ["DEFECT-SENTINEL"]}
    ],
    "sourceInspectionIds": [987654321, 987654322],
    "finalLabel": "REJECT",
    "inspectionStatus": "COMPLETED",
    "failureType": None,
    "failureReason": None,
}


def test_all_request_fields_reach_individual_report_prompt():
    request = IndividualReportRequest(**REQUEST_DATA)
    retained = request.model_dump()

    assert set(retained) == {
        "cellSerialNo",
        "inspectionId",
        "totalImages",
        "cellSize",
        "pointGroups",
        "ctVoidRatio",
        "rgbDefectRate",
        "defectInfo",
        "sourceInspectionIds",
        "finalLabel",
        "inspectionStatus",
        "failureType",
        "failureReason",
    }

    _, _, prompt = build_individual_report_prompt(retained)
    expected_prompt_values = {
        "cellSerialNo": "SERIAL-PROMPT-8",
        "inspectionId": "987654321",
        "totalImages": "37",
        "cellSize": "[101.25, 202.5, 7.75]",
        "pointGroups": "[[303.75, 404.5], [505.25, 606.0]]",
        "ctVoidRatio": "0.123456",
        "rgbDefectRate": "87.654321",
        "defectInfo": "DEFECT-SENTINEL",
        "sourceInspectionIds": "987654322",
        "finalLabel": "REJECT",
        "inspectionStatus": "COMPLETED",
    }
    for field, value in expected_prompt_values.items():
        assert value in prompt, f"{field} no longer reaches the report prompt"


def test_individual_report_request_rejects_unknown_contract_fields():
    with pytest.raises(ValidationError):
        IndividualReportRequest(**REQUEST_DATA, unexpectedContractField="must fail")
