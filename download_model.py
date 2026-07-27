import getpass
import os
import sys

import torch
from dotenv import load_dotenv
from huggingface_hub import get_token, login
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

load_dotenv()

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def authenticate():
    # 1순위: ~/.cache/huggingface/token (이전 로그인 세션)
    if get_token() is not None:
        print("✅ 저장된 토큰으로 자동 로그인")
        return

    # 2순위: .env 파일의 HF_TOKEN
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        login(token=env_token)
        print("✅ .env 토큰으로 로그인 완료")
        return

    # 3순위: 직접 입력
    token = getpass.getpass("HuggingFace 토큰 입력: ")
    login(token=token)
    print("✅ 로그인 완료")


def download_model():
    print(f"\n📦 모델 다운로드 시작: {MODEL_ID}")
    print("   (첫 실행 시 수 GB 다운로드 — 시간이 걸릴 수 있습니다)\n")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    print("✅ 모델 로드 완료")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("✅ 프로세서 로드 완료")

    return model, processor


def verify(model, processor):
    print("\n🔍 동작 검증 중...")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": "안녕하세요. 정상 작동 확인용 메시지입니다."}]},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    print(f"✅ 모델 응답: {response.strip()}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다. GPU 및 드라이버를 확인하세요.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)} GB\n")

    authenticate()
    model, processor = download_model()
    verify(model, processor)

    print("\n🎉 모델 캐싱 완료. 이후 실행부터는 캐시에서 즉시 로드됩니다.")
    print(f"   캐시 위치: ~/.cache/huggingface/hub/")
