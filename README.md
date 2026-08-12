# kt-aivle-big-proj-vlm-report
KT AIVLE 빅프로젝트 16조 — vLM 서버(LangGraph + Qwen)

develop과는 다르게 test 브랜치는 현재 Qwen3-VL-4B-Instruct 사용 중. 모델 설치, 호출 부분 차이.

## 📖 프로젝트 개요
본 프로젝트는 배터리 셀 품질 검사 데이터를 바탕으로 **시각 언어 모델(VLM, Qwen3-VL)**과 **LangGraph**를 활용하여 공정 품질 진단 보고서를 자동 생성하고 이미지 화질을 검사하는 FastAPI 기반의 AI 서버입니다. 
AI 모델의 환각(Hallucination) 현상을 방지하기 위해 LangGraph 기반의 **자가 검수(Critic) 파이프라인**을 구축하여 신뢰성 높은 결과물을 제공합니다.

---

## ✨ 주요 기능

### 1. 일일 품질 종합 보고서 생성 (`POST /vlm/reports/daily`)
- 하루 동안 발생한 공정 데이터(PASS/REJECT 수량, 수율, 주요 결함 유형)를 종합하여 **일일 생산 및 수율 요약 마크다운 리포트**를 생성합니다.
- **Critic 검수 로직**: 수치 정확성, 수율 계산(%), 섹션 누락, 원본 데이터에 없는 날조(Hallucination) 여부를 검증하며 실패 시 모델이 피드백을 반영하여 최대 1회 재작성합니다.

### 2. 개별 셀 불량 분석 보고서 생성 (`POST /vlm/reports/individual`)
- 개별 배터리 셀의 검사 데이터(CT/RGB 이미지 수, 공극체적률, 표면 불량면적 결합률 등)를 기반으로 **개별 검사 리포트**를 생성합니다.
- 결함의 위치적 특성 분석 및 주요 결함 유형을 코멘트합니다.

### 3. 이미지 품질 검사 (`POST /vlm/qualityInspection`)
- **하이브리드 검사 파이프라인**: 
  1. **OpenCV (Rule-based)**: Laplacian(초점 불량), 평균 밝기(노출 과다/부족), GaussianBlur 기반 노이즈 레벨 측정 등 1차 물리적 화질 검증을 수행합니다.
  2. **VLM (Qwen3-VL)**: 1차 검증을 통과한 이미지에 한해, 심층적인 시각적 결함(실오라기, 반사광, 모션 아티팩트, 줄무늬 왜곡 등)을 분석하고 판별 기준 매핑 후 JSON 결과를 반환합니다.
- **에러 서비스**: CT 및 RGB 특성에 맞는 오류 원인을 판별하고 분류합니다.

---

## 🛠 기술 스택

- **Web Framework**: FastAPI, Uvicorn (Python 3.12)
- **AI & VLM**: `Qwen/Qwen3-VL-4B-Instruct` (4-bit Quantization, bfloat16 지원)
- **LLM Orchestration**: LangGraph, LangChain
- **Computer Vision**: OpenCV (`opencv-python-headless`), NumPy
- **Infrastructure**: CUDA 12.4, Docker, Kubernetes
- **Hardware Requirement**: NVIDIA RTX 4070 Ti (12GB VRAM) 이상 권장

---

## 🏗 시스템 아키텍처 및 파이프라인 (LangGraph)

프로젝트는 `StateGraph`를 활용하여 각 서비스의 워크플로우를 오케스트레이션합니다.

### 📊 Report Service 워크플로우 (일일 & 개별 리포트)
```text
[START] 
  └──> 생성 노드 (Report Node: Qwen 모델로 리포트 초안 생성)
         └──> 검수 노드 (Critic Node: 수치 일치, 날조 여부, 양식 완전성 검증)
                ├──> [PASS] ────> [END] (최종 리포트 반환)
                └──> [FAIL] ────> (최대 1회 재시도, Critic 피드백 반영) ────> 생성 노드 재진입
```

### 🖼️ Image Quality Inspection 워크플로우
```text
[START]
  └──> 물리적 화질 검사 (cv_inspection_node: OpenCV 기반 블러, 밝기, 노이즈 확인)
         ├──> [불합격] ────> [END] (CV 에러 반환)
         └──> [합격] ──────> VLM 심층 검사 (image_quality_inspection_node: Qwen으로 시각 결함 판별)
                               └──> [END]
```

---

## 🚀 설치 및 실행 가이드

### 1. 환경 변수 설정
루트 경로에 `.env` 파일을 생성하고 HuggingFace 토큰을 입력합니다.
```env
HF_TOKEN=your_huggingface_api_token
```

### 2. 로컬 실행 (Python 가상환경)
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 패키지 설치 (CUDA 12.4용 PyTorch 및 Transformers 최신 버전 포함)
pip install -r requirements.txt

# VLM 모델 사전 다운로드 및 캐싱 (최초 1회 필수)
python download_model.py

# FastAPI 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### 3. Docker 배포
```bash
# 도커 이미지 빌드
docker build -t vlm-report-server .

# 도커 컨테이너 실행 (GPU 할당, 모델 캐시 볼륨 마운트)
docker run -d \
  --gpus all \
  -p 7860:7860 \
  -e HF_TOKEN="your_huggingface_api_token" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name vlm-server \
  vlm-report-server
```

---

## 🌐 API 명세 (Endpoint Specs)

| Method | Endpoint | 설명 |
| :--- | :--- | :--- |
| `GET` | `/health` | 서버 헬스 체크 및 가동 상태 확인 |
| `POST` | `/vlm/reports/daily` | 일일 공정 데이터를 바탕으로 종합 진단 리포트(Markdown) 생성 |
| `POST` | `/vlm/reports/individual` | 개별 배터리 셀의 결함 데이터를 분석하여 개별 리포트 생성 |
| `POST` | `/vlm/qualityInspection` | 배열로 전달된 이미지 목록의 화질 오류 원인 및 결함(CT/RGB) 판독 |

*(상세한 Request Payload 및 Response 예시는 `docs/API_SPEC/` 디렉토리 내의 문서를 참고하세요.)*

---

## 📁 디렉토리 구조
```text
kt-aivle-big-proj-vlm/
├── app/
│   ├── api/routes/          # FastAPI 라우터 (reports, imageQuality, health)
│   ├── clients/             # vLLM 및 HuggingFace 클라이언트 (모델 로드/추론)
│   ├── graph/               # LangGraph 노드, 상태(State) 및 엣지 관리
│   │   ├── daily_report_service/
│   │   ├── individual_report_service/
│   │   ├── image_quality_inspection_service/
│   │   └── error_service/
│   ├── schemas/             # Pydantic 기반 Request/Response 스키마
│   └── services/            # API 요청과 LangGraph 워크플로우를 연결하는 서비스 레이어
├── docs/                    # API 설계 명세서 (Markdown)
├── k8s/                     # Kubernetes 배포 파일 (deployment, service, configmap)
├── download_model.py        # Qwen 모델 초기 다운로드 및 정상 동작 검증 스크립트
├── colab_run.ipynb          # Google Colab(T4 GPU) 구동 및 ngrok 터널링 테스트용 노트북
├── Dockerfile               # 도커 이미지 빌드 스크립트
└── requirements.txt         # 파이썬 의존성 패키지 목록
```
