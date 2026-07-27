import asyncio
from app.clients.vllm_client import load_model
from app.schemas.request import MOCK_DAILY_REPORT_REQUEST
from app.services.report_service import generate_daily_report


async def main():
    print("모델 로딩 중...")
    load_model()

    print("리포트 생성 중...")
    result = await generate_daily_report(MOCK_DAILY_REPORT_REQUEST)

    print("\n===== 결과 =====")
    print(f"제목: {result.title}")
    print(f"크리틱 판정: {result.critic_verdict}")
    print(f"재시도 횟수: {result.retry_count}")
    print(f"\n{result.generated_report}")


if __name__ == "__main__":
    asyncio.run(main())
