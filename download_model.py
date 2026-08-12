"""가중치를 미리 받아 캐시에 넣는다.

기동 시점의 최초 다운로드는 수 GB 라 헬스체크 유예시간을 넘긴다. 배포 호스트에서
컨테이너를 띄우기 전에 한 번 돌려 캐시를 채워 두는 용도다.

모델 이름과 정밀도는 app.core.config 가 정한다. 이 파일에 다시 박지 않는다.
서비스와 다른 값으로 받아 두면 캐시가 있어도 기동 때 다시 받는다.
"""
import getpass
import sys

import torch
from dotenv import load_dotenv
from huggingface_hub import get_token, login
from transformers import AutoModelForMultimodalLM, AutoProcessor

from app.clients.vllm_client import _TORCH_DTYPE
from app.core.config import load_settings

load_dotenv()


def authenticate(settings) -> None:
    if get_token() is not None:
        print("저장된 토큰으로 자동 로그인")
        return

    if settings.hf_token:
        login(token=settings.hf_token)
        print("환경변수 토큰으로 로그인 완료")
        return

    token = getpass.getpass("HuggingFace 토큰 입력: ")
    login(token=token)
    print("로그인 완료")


def download(settings):
    print(f"\n모델 다운로드 시작: {settings.model_id}")
    print("   (첫 실행 시 수 GB 다운로드 — 시간이 걸릴 수 있습니다)\n")

    load_kwargs = {"dtype": _TORCH_DTYPE[settings.dtype]}
    if settings.device == "cuda":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForMultimodalLM.from_pretrained(
        settings.model_id,
        **load_kwargs,
    )
    print("모델 로드 완료")

    processor = AutoProcessor.from_pretrained(settings.model_id)
    print("프로세서 로드 완료")

    return model, processor


def verify(model, processor) -> None:
    print("\n동작 검증 중...")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "안녕하세요. 정상 작동 확인용 메시지입니다."},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    print(f"모델 응답: {response.strip()}")


if __name__ == "__main__":
    settings = load_settings()

    # DEVICE=cpu 로도 캐시는 채울 수 있다. GPU 가 없는 개발자가 가중치만
    # 미리 받아 두는 경우를 막지 않는다.
    if settings.device == "cuda" and not torch.cuda.is_available():
        print("DEVICE=cuda 인데 CUDA 를 쓸 수 없습니다. 드라이버를 확인하거나 DEVICE=cpu 로 실행하세요.")
        sys.exit(1)

    if settings.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"VRAM: {round(total_memory / 1e9, 1)} GB\n")

    authenticate(settings)
    model, processor = download(settings)
    verify(model, processor)

    print("\n모델 캐싱 완료. 이후 실행부터는 캐시에서 즉시 로드됩니다.")
    print("   캐시 위치: HF_HOME 이 가리키는 경로 (기본 ~/.cache/huggingface)")
