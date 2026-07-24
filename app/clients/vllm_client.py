import asyncio
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

hf_model = None
hf_processor = None


def load_model(model_id: str = "Qwen/Qwen3-VL-4B-Instruct") -> None:
    global hf_model, hf_processor

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    hf_processor = AutoProcessor.from_pretrained(model_id)


async def invoke_qwen_hf(
    system_msg: str,
    prompt_text: str,
    image_urls: list[str] = None,
) -> str:
    content = []

    if image_urls:
        for url in image_urls:
            content.append({"type": "image", "image": url, "max_pixels": 1003520})
    content.append({"type": "text", "text": prompt_text})

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": content},
    ]

    text = hf_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = hf_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    def generate():
        return hf_model.generate(
            **inputs, max_new_tokens=1500, temperature=0.2, do_sample=True
        )

    generated_ids = await asyncio.to_thread(generate)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = hf_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]
