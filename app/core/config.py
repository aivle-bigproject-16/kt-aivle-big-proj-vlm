"""런타임 설정.

값은 전부 환경변수에서 온다. 이름은 인프라 레포의 compose.yaml 이 주입하는 것과
같아야 한다. 이름이 어긋나면 값이 조용히 무시되고 기본값으로 떨어진다 —
그 사고가 이미 한 번 있었다(compose 가 VLM_MODEL_ID 를 주는데 코드는
MODEL_ID 를 하드코딩하고 있었다).

ai-infer 레포의 app/settings.py 와 같은 방식으로 쓴다. 검증은 기동 시점에 하고,
잘못된 값이면 그 자리에서 예외를 던진다. 추론 중간에 드러나면 원인 추적이 어렵다.
"""
import os
from dataclasses import dataclass


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"


@dataclass(frozen=True)
class Settings:
    model_id: str
    device: str
    dtype: str
    quantization: str
    hf_token: str | None


def _validated(name: str, value: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise ValueError(
            f"{name} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


def load_settings() -> Settings:
    # 모델 교체는 이 값으로만 한다. 코드에 모델 이름을 다시 박지 않는다.
    model_id = os.getenv("VLM_MODEL_ID", "").strip() or DEFAULT_MODEL_ID

    device = _validated(
        "DEVICE",
        os.getenv("DEVICE", "cuda").strip().lower(),
        {"cuda", "cpu"},
    )

    # 배포 GPU 는 L40S 다. bf16 을 지원하므로 fp16 으로 묶을 이유가 없다.
    # 계약이 fp16 을 기본으로 적어 둔 것은 T4 를 전제하던 시절의 결정이다.
    dtype = _validated(
        "VLM_DTYPE",
        os.getenv("VLM_DTYPE", "bf16").strip().lower(),
        {"fp16", "bf16", "fp32"},
    )

    quantization = _validated(
        "VLM_QUANTIZATION",
        os.getenv("VLM_QUANTIZATION", "none").strip().lower(),
        {"none", "4bit", "8bit"},
    )

    # HUGGING_FACE_HUB_TOKEN 은 huggingface_hub 이 스스로 읽는 표준 이름이고
    # compose 가 주는 이름이다. HF_TOKEN 은 이 레포가 로컬에서 쓰던 이름이라
    # 기존 .env 를 쓰는 팀원이 깨지지 않도록 함께 받는다.
    hf_token = (
        os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or None
    )

    return Settings(
        model_id=model_id,
        device=device,
        dtype=dtype,
        quantization=quantization,
        hf_token=hf_token,
    )
