# syntax=docker/dockerfile:1
#
# 생성 모델 서버 이미지.
#
#   docker build -t battery-vlm .
#
# 베이스는 ai-infer 의 GPU 이미지와 같다. 두 컨테이너가 EC2 한 대에서 GPU 한 장을
# 나눠 쓰므로, CUDA·cuDNN 계열이 갈리면 한쪽이 드라이버와 어긋난다.
# 이 이미지가 torch 를 이미 갖고 있어서 requirements.txt 에는 torch 가 없다.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# 포트는 계약값 8001 이다. compose 가 backend-ai 를 이 주소로 붙인다.
ENV VLM_PORT=8001 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 가중치 캐시. compose 가 vlm-cache 볼륨을 이 경로에 붙인다.
# 값이 다르면 컨테이너를 다시 만들 때마다 수 GB 를 다시 받는다.
ENV HF_HOME=/cache/huggingface

# 의존성을 먼저 넣는다. 소스만 바뀌었을 때 설치 단계를 건너뛰기 위함이다.
# GPU 이미지는 설치가 오래 걸려 이 순서가 특히 중요하다.
# 베이스의 python 은 배포판이 관리하는 환경(PEP 668)이라 플래그 없이 pip 를 쓰면
# externally-managed-environment 로 즉시 실패한다. ai-infer 의 GPU Dockerfile 도
# 같은 이유로 --break-system-packages 를 쓰고 있으니 두 레포의 관용구를 맞춘다.
COPY requirements.txt ./requirements.txt
RUN python -m pip install \
      --break-system-packages \
      --no-cache-dir \
      --timeout 1000 \
      -r requirements.txt

COPY app ./app
COPY download_model.py ./download_model.py

# 루트로 돌리지 않는다. 캐시 경로는 이 사용자가 쓸 수 있어야 한다.
# 베이스 이미지에 이미 UID 1000(ubuntu)이 있어서 같은 UID 로 useradd 를 하면
# "UID 1000 is not unique" 로 빌드가 깨진다. 새로 만들지 않고 그 UID 를 그대로 쓴다.
# 이름 대신 숫자를 쓰는 것은 베이스가 사용자 이름을 바꿔도 깨지지 않게 하기 위함이다.
RUN mkdir -p /cache/huggingface \
 && chown -R 1000:1000 /app /cache
USER 1000:1000

EXPOSE 8001

# 헬스체크는 compose 가 정의한다(GET /health, start_period 300s).
# 최초 기동은 가중치 다운로드가 포함돼 오래 걸린다. 미리 받아 두려면
# 호스트에서 python download_model.py 를 한 번 돌린다.
#
# --workers 를 주지 않는다. 워커마다 모델을 통째로 한 벌씩 올려 VRAM 이 배로 든다.
# 동시 실행 상한은 BE 의 LLM_POOL_SIZE 가 잡는다.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${VLM_PORT}"]
