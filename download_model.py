import getpass
import os
import sys

import torch
from dotenv import load_dotenv
from huggingface_hub import get_token, login
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_ID = "Qwen/Qwen3.5-2B"


def authenticate():
    if get_token() is not None:
        print("✅ 저장된 토큰으로 자동 로그인")
        return

    env_token = os.getenv("HF_TOKEN")
    if env_token:
        login(token=env_token)
        print("✅ .env 토큰으로 로그인 완료")
        return

    token = getpass.getpass("HuggingFace 토큰 입력: ")
    login(token=token)
    print("✅ 로그인 완료")


def download_model():
    print(f"\n📦 모델 다운로드 시작: {MODEL_ID}")
    print("   (첫 실행 시 수 GB 다운로드 — 시간이 걸릴 수 있습니다)\n")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("✅ 모델 로드 완료")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("✅ 토크나이저 로드 완료")

    return model, tokenizer


def verify(model, tokenizer):
    print("\n🔍 동작 검증 중...")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "안녕하세요. 정상 작동 확인용 메시지입니다."},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    trimmed = output_ids[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]

    print(f"✅ 모델 응답: {response.strip()}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다. GPU 및 드라이버를 확인하세요.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)} GB\n")

    authenticate()
    model, tokenizer = download_model()
    verify(model, tokenizer)

    print("\n🎉 모델 캐싱 완료. 이후 실행부터는 캐시에서 즉시 로드됩니다.")
    print(f"   캐시 위치: ~/.cache/huggingface/hub/")
