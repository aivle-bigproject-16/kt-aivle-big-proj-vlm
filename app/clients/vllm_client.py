import asyncio
import time
import torch
import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

hf_model = None
hf_processor = None

# 🚨 주의: 이미지를 처리하려면 텍스트 전용(Qwen3.5)이 아닌 VL 모델을 사용해야 합니다.
# 사양에 맞춰 "Qwen/Qwen2-VL-2B-Instruct" 또는 "Qwen/Qwen2.5-VL-3B-Instruct" 등을 사용하세요.
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

def load_model(model_id: str = MODEL_ID) -> None:
    global hf_model, hf_processor
    
    # 모델 로드
    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # 🚨 VLM(비전-언어 모델)은 Tokenizer 대신 Processor를 사용해야 합니다.
    hf_processor = AutoProcessor.from_pretrained(model_id)
    print(f"✅ {model_id} 로드 완료")


async def invoke_qwen_hf(
    system_msg: str,
    prompt_text: str,
    image_urls: list[str] = None,
) -> str:
    if image_urls is None:
        image_urls = []

    # 1. 메시지 content 리스트 구성 (이미지와 텍스트 혼합)
    user_content = []
    
    # 이미지 경로가 리스트로 들어왔을 경우 각각 처리
    for img_path in image_urls:
        abs_path = os.path.abspath(img_path)
        # 로컬 경로인 경우 file:// 스키마 추가, 웹 URL인 경우 그대로 사용
        image_uri = f"file://{abs_path}" if not img_path.startswith(("http", "file://")) else img_path
        
        # 모델이 인식할 수 있게 type과 image 키를 사용
        user_content.append({"type": "image", "image": image_uri})
        
    # 마지막으로 사용자의 텍스트 프롬프트 추가
    user_content.append({"type": "text", "text": prompt_text})

    # 최종 메시지 구조 완성
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]

    # 2. 챗 템플릿 적용 (텍스트 부분 구조화)
    text = hf_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 3. 비전 데이터 전처리 (이미지 파일들을 텐서로 바꿀 준비)
    image_inputs, video_inputs = process_vision_info(messages)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 4. Processor를 통해 최종 입력 텐서 생성 (텍스트 + 이미지)
    inputs = hf_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    def generate():
        return hf_model.generate(
            **inputs, max_new_tokens=1500, temperature=0.2, do_sample=True
        )

    print("🔄 Qwen 추론 시작...", flush=True)
    t0 = time.perf_counter()
    generated_ids = await asyncio.to_thread(generate)
    print(f"✅ Qwen 추론 완료 — {time.perf_counter() - t0:.1f}초", flush=True)

    # 5. 결과 디코딩 (입력 프롬프트 부분을 제외하고 생성된 답변만 추출)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = hf_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]