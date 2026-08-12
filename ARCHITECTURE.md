# 프로젝트 아키텍처

## 개발 환경

| 항목 | 내용 |
|---|---|
| OS (호스트) | Windows 11 Pro |
| 실행 환경 | WSL2 Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 4070 Ti (12GB VRAM) |
| CUDA 드라이버 | 591.86 (CUDA 13.1 지원) |
| Python | 3.12 |
| 패키지 관리 | pip + venv (`.venv/`) |
| 에디터 | VS Code (WSL Remote 확장) |

---

## 프로젝트 구조

```
kt-aivle-big-proj-vlm/
│
├── app/
│   ├── main.py                        # FastAPI 앱 진입점
│   │
│   ├── api/routes/
│   │   ├── reports.py                 # POST /reports/{id}/generate
│   │   └── health.py                  # GET /health
│   │
│   ├── core/
│   │   ├── config.py                  # 환경 설정 (Pydantic Settings)
│   │   └── logging.py
│   │
│   ├── queue/
│   │   ├── job_queue.py               # asyncio.Queue 래퍼
│   │   └── worker.py                  # worker_loop()
│   │
│   ├── graph/
│   │   ├── report_service/            # 일일 보고서 그래프
│   │   │   ├── state.py               # ReportState (TypedDict)
│   │   │   ├── edges.py               # route_after_critic()
│   │   │   ├── graph_builder.py       # StateGraph 조립 + compile()
│   │   │   └── nodes/
│   │   │       ├── daily_report.py    # 보고서 생성 노드
│   │   │       └── critic.py         # 검수 노드
│   │   │
│   │   └── error_service/             # 미구현 (예정)
│   │
│   ├── clients/
│   │   ├── vllm_client.py             # HuggingFace 모델 로드 + 추론
│   │   ├── storage_client.py          # S3 등 스토리지
│   │   └── backend_client.py          # 완료 후 백엔드 콜백 POST
│   │
│   ├── schemas/
│   │   ├── request.py
│   │   └── response.py
│   │
│   └── services/
│       └── report_service.py          # 큐 적재 + graph 오케스트레이션
│
├── tests/
│   ├── test_graph.py
│   └── test_api.py
│
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
│
├── download_model.py                  # 모델 초기 다운로드 및 검증
├── daily_report_graph.ipynb           # 그래프 독립 실행 버전 (개발/테스트용)
├── requirements.txt
├── Dockerfile
├── .env                               # 환경 변수 (gitignore)
└── .env.example                       # 환경 변수 템플릿
```

---

## VLM 모델

| 항목 | 내용 |
|---|---|
| 모델 | Qwen/Qwen3-VL-4B-Instruct |
| 양자화 | 4-bit (bitsandbytes nf4, double quant) |
| dtype | bfloat16 |
| 디바이스 | CUDA (device_map="auto") |
| 예상 VRAM | ~3.8 GB |
| 캐시 위치 | `~/.cache/huggingface/hub/` (WSL2) |
| 추론 방식 | asyncio.to_thread() — 비동기 블로킹 방지 |

---

## LangGraph — report_service

### State

```python
class ReportState(TypedDict):
    daily_data: Optional[DailyData]   # 입력 데이터
    generated_report: str             # 생성된 보고서 (마크다운)
    title: str                        # 보고서 제목
    retry_count: int                  # 재시도 횟수
    critic_verdict: Optional[str]     # "PASS" | "FAIL"
    critic_issues: Optional[List[dict]]  # 검수 이슈 목록
```

### 그래프 흐름

```
START
  └─→ daily_node (보고서 생성)
        └─→ critic_node (검수)
              ├─ PASS ──────────────→ END
              └─ FAIL (retry ≤ 1) → daily_node (critic_issues 반영하여 재생성)
                    └─→ critic_node
                          └─────────→ END (PASS/FAIL 무관)
```

### 노드별 역할

| 노드 | 파일 | 역할 |
|---|---|---|
| `daily_node` | `nodes/daily_report.py` | 일일 품질 종합 보고서 생성 (마크다운) |
| `critic_node` | `nodes/critic.py` | 수치 정확성·섹션 완전성·날조 여부 검수, JSON 반환 |

### 크리틱 검수 기준

1. 섹션 완전성 (4개 헤더 존재 여부)
2. 수치 일치 (totalCount, passCount, rejectCount, failedCount)
3. 수율 계산 정확성 (`round(passCount / totalCount * 100, 1)`)
4. 결함 순위 (count 내림차순)
5. 제조사 순위 (count 내림차순)
6. 날조 금지 (원본에 없는 데이터 생성 여부)

## LangGraph — Image Quality Inspection service

### State
```python
class QualityState(TypedDict):
    imageType: Optional[str]
    images: List[Images]
    vlm_target_images: List[Images] 
    
    inspection_result: Annotated[List[Dict[str, Any]], operator.add]
```

### 그래프 흐름
```text
[START]
  └──> 물리적 화질 검사 (cv_inspection_node: OpenCV 기반 블러, 밝기, 노이즈 확인)
         ├──> [불합격] ────> [END] (CV 에러 반환)
         └──> [합격] ──────> VLM 심층 검사 (image_quality_inspection_node: Qwen으로 시각 결함 판별)
                               └──> [END]
```

### 노드별 역할

| 노드 | 파일 | 역할 |
|---|---|---|
| `cv_inspection_node` | `nodes/cv_inspection_node.py` | 일차적 규칙 기반 분석 결과|
| `image_quality_inspection_node` | `nodes/image_quality_inspection_node.py` | VLM을 통한 이미지 분석 결과 |

---

## 환경 변수 (.env)

| 키 | 설명 |
|---|---|
| `HF_TOKEN` | HuggingFace API 토큰 |

---

## 의존성 주요 패키지

| 분류 | 패키지 |
|---|---|
| API 서버 | FastAPI, uvicorn, pydantic-settings |
| VLM 추론 | torch (cu124), transformers, bitsandbytes, accelerate, qwen-vl-utils |
| 그래프 | langgraph, langchain-core |
| 스토리지 | boto3 |
| 환경 관리 | python-dotenv |
|openCV 추론| pillow, requests, numpy, opencv-python-headless, aiohttp|

---

## 배포 (예정)

| 항목 | 내용 |
|---|---|
| 컨테이너 | Docker |
| 오케스트레이션 | Kubernetes (k8s/) |
| GPU 노드 | CUDA 지원 노드 필요 |
