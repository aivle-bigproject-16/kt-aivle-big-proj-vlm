import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app

client = TestClient(app)

# 1. 헬스 체크 엔드포인트 테스트
@patch("app.api.routes.health.is_model_loaded", return_value=True)
def test_health_check(mock_is_model_loaded):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# 2. 일일 리포트 API 스키마 계약 테스트
@patch("app.services.daily_report_service.report_llm_model.ainvoke", new_callable=AsyncMock)
def test_daily_report_api_schema(mock_ainvoke):
    # 그래프 실행 결과를 가짜(Mock)로 설정
    mock_ainvoke.return_value = {
        "title": "2026-08-14 일일 보고서",
        "generated_report": "### 테스트 보고서"
    }
    
    payload = {
        "daily_data": {
            "reportDate": "2026-08-14",
            "summaryData": {
                "totalCount": 100, "passCount": 90, "rejectCount": 10, "failedCount": 0,
                "prevTotalCount": 100, "prevRejectCount": 10, "defects": []
            }
        }
    }
    
    response = client.post("/vlm/reports/daily", json=payload)
    
    # 응답 규격(Schema) 검증
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "COMPLETED"
    assert "2026-08-14" in data["title"]
    assert "테스트 보고서" in data["content"]


# 3. 이미지 품질 검사 API 스키마 계약 테스트
@patch("app.services.image_quality_service.image_inspection_graph.ainvoke", new_callable=AsyncMock)
def test_quality_inspection_api_schema(mock_ainvoke):
    mock_ainvoke.return_value = {
        "inspection_result": [
            {"imageId": "img-001", "failType": "rgb_focus_failure", "description": "초점 불량 테스트"}
        ]
    }
    
    payload = {
        "imageType": "RGB",
        "images": [{"imageId": "img-001", "imageUrl": "http://dummy.url/img.jpg"}]
    }
    
    response = client.post("/vlm/qualityInspection", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "COMPLETED"
    assert data["content"][0]["failType"] == "rgb_focus_failure"