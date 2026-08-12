# 1. Base Image: 프로젝트 환경에 맞춘 Python 3.12 슬림 버전
FROM python:3.12-slim

# 2. 시스템 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 필수 시스템 패키지 설치
# git: requirements.txt 내 transformers git 레포지토리 설치용
# libglib2.0-0: opencv-python-headless 구동용 필수 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. 의존성 파일 복사 및 패키지 설치
# 캐시를 활용하여 빌드 속도를 높이기 위해 requirements.txt만 먼저 복사합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 프로젝트 소스 코드 복사
COPY . .

# 7. 포트 노출 (Colab 테스트 환경의 7860 포트 또는 FastAPI 기본 8000 포트)
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# 8. 컨테이너 실행 시 FastAPI 서버 구동
# 필요에 따라 포트 번호를 8000 등으로 수정할 수 있습니다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]