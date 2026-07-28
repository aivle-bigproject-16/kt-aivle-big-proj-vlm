import asyncio
from app.clients.vllm_client import load_model
from app.schemas.request import MOCK_DAILY_REPORT_REQUEST
from app.services.daily_report_service import generate_daily_report


async def main():
    print("모델 로딩 중...")
    load_model()

    print("리포트 생성 중...")
    result = await generate_daily_report(MOCK_DAILY_REPORT_REQUEST)

    print("\n===== 결과 =====")
    print(f"상태: {result.status}")
    print(f"제목: {result.title}")
    print(f"\n{result.content or result.failureReason}")


if __name__ == "__main__":
    asyncio.run(main())
