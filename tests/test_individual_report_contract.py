import pytest
from pydantic import ValidationError

from app.graph.individual_report_service.nodes.individual_report_node import (
    build_individual_report_prompt,
)
from app.graph.individual_report_service.report_quality import (
    check_failure_reinspection,
    check_korean_only,
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


def test_prompt_forbids_chinese_text_and_critic_rejects_leakage():
    _, system_message, prompt = build_individual_report_prompt(REQUEST_DATA)

    assert "중국어와 한자를 섞지 않고" in system_message
    assert "중국어 또는 한자를 사용하지 않습니다" in prompt
    assert check_korean_only("결함이 발견되지 않았습니다.") == []
    assert check_korean_only("결함 패턴이没有出现.") == [{
        "criterion": 6,
        "description": "한국어 리포트에 중국어 한자가 포함됨: 出有没现",
    }]


def test_fail_report_requires_explicit_reinspection_guidance():
    assert check_failure_reinspection("추가 확인이 필요합니다.", {"finalLabel": "PASS"}) == []
    assert check_failure_reinspection("재검사를 권고합니다.", {"finalLabel": "FAIL"}) == []
    assert check_failure_reinspection("추가 확인이 필요합니다.", {"finalLabel": "FAIL"}) == [{
        "criterion": 7,
        "description": "FAIL 리포트에 재검사 권고가 누락됨",
    }]
