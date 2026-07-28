import asyncio
import logging
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

hf_model = None
hf_tokenizer = None

MODEL_ID = "Qwen/Qwen3.5-2B"


def load_model(model_id: str = MODEL_ID) -> None:
    global hf_model, hf_tokenizer

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"✅ {model_id} 로드 완료")


async def invoke_qwen_hf(
    system_msg: str,
    prompt_text: str,
    image_urls: list[str] = None,
) -> str:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_text},
    ]

    text = hf_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = hf_tokenizer(text, return_tensors="pt").to("cuda")

    def generate():
        return hf_model.generate(
            **inputs, max_new_tokens=1500, temperature=0.2, do_sample=True
        )

    logger.info("🔄 Qwen 추론 시작...")
    t0 = time.perf_counter()
    generated_ids = await asyncio.to_thread(generate)
    logger.info("✅ Qwen 추론 완료 — %.1f초", time.perf_counter() - t0)

    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    output_text = hf_tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]
