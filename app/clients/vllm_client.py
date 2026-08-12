"""생성 모델 클라이언트.

모델 카드가 지정하는 조합은 AutoProcessor + AutoModelForMultimodalLM 이다.
이전 구현은 AutoModelForCausalLM + AutoTokenizer 를 썼는데, 체크포인트의
아키텍처가 Qwen3_5ForConditionalGeneration 이라 그 조합으로는 적재되지 않는다.
load_model() 은 FastAPI lifespan 에서 불리므로, 실패하면 앱이 아예 뜨지 않는다.

프로세서를 쓰면 이미지도 같은 경로로 들어간다. 텍스트만 보낼 때는 content 를
문자열로, 이미지를 함께 보낼 때는 content 를 블록 배열로 만든다.
"""
import asyncio
import time

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from app.core.config import Settings, load_settings

hf_model = None
hf_processor = None
_settings: Settings | None = None

_TORCH_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _quantization_config(settings: Settings):
    """양자화 설정. none 이면 None 을 준다.

    BitsAndBytesConfig 는 bitsandbytes 가 있어야 만들어지고 CUDA 를 전제한다.
    CPU 로 돌릴 때 4bit 를 요구하면 적재 도중에 터지므로 여기서 먼저 막는다.
    """
    if settings.quantization == "none":
        return None

    if settings.device != "cuda":
        raise ValueError(
            "VLM_QUANTIZATION requires DEVICE=cuda, "
            f"got DEVICE={settings.device}"
        )

    from transformers import BitsAndBytesConfig

    if settings.quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=_TORCH_DTYPE[settings.dtype],
    )


def load_model(settings: Settings | None = None) -> None:
    global hf_model, hf_processor, _settings

    _settings = settings or load_settings()

    # transformers 5.x 의 인자 이름은 dtype 이다. torch_dtype 도 아직 받지만
    # 호출할 때마다 deprecation 경고를 남긴다.
    load_kwargs = {
        "dtype": _TORCH_DTYPE[_settings.dtype],
    }

    quantization_config = _quantization_config(_settings)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    if _settings.device == "cuda":
        # 가중치를 가용 장치에 나눠 싣는다. CPU 로 돌릴 때 이 값을 주면
        # accelerate 가 개입해 오히려 느려지므로 GPU 일 때만 쓴다.
        load_kwargs["device_map"] = "auto"

    if _settings.hf_token:
        load_kwargs["token"] = _settings.hf_token

    hf_model = AutoModelForMultimodalLM.from_pretrained(
        _settings.model_id,
        **load_kwargs,
    )

    if _settings.device == "cpu":
        hf_model = hf_model.to("cpu")

    hf_processor = AutoProcessor.from_pretrained(
        _settings.model_id,
        token=_settings.hf_token,
    )

    print(
        f"모델 적재 완료 — {_settings.model_id} "
        f"(device={_settings.device}, dtype={_settings.dtype}, "
        f"quantization={_settings.quantization})",
        flush=True,
    )


def _build_messages(
    system_msg: str,
    prompt_text: str,
    image_urls: list[str] | None,
) -> list[dict]:
    """이미지가 있으면 블록 배열로, 없으면 문자열로 content 를 만든다.

    이미지 블록을 텍스트보다 앞에 둔다. 모델 카드의 예시 순서이고, 질문이
    이미지를 뒤이어 가리키는 형태라 프롬프트가 자연스럽다.
    """
    if not image_urls:
        user_content = prompt_text
    else:
        user_content = [
            {"type": "image", "url": url} for url in image_urls
        ]
        user_content.append({"type": "text", "text": prompt_text})

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


async def invoke_qwen_hf(
    system_msg: str,
    prompt_text: str,
    image_urls: list[str] | None = None,
) -> str:
    if hf_model is None or hf_processor is None:
        raise RuntimeError(
            "model is not loaded; load_model() runs in the app lifespan"
        )

    messages = _build_messages(system_msg, prompt_text, image_urls)

    inputs = hf_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(hf_model.device)

    def generate():
        return hf_model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.2,
            do_sample=True,
        )

    print("생성 시작", flush=True)
    started_at = time.perf_counter()
    generated_ids = await asyncio.to_thread(generate)
    print(
        f"생성 완료 — {time.perf_counter() - started_at:.1f}초",
        flush=True,
    )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, prompt_length:]

    output_text = hf_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]
