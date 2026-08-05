import os
import chromadb
import torch
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 1. 임베딩 용 CLIP 모델 및 프로세서 설정
model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
embedding_model = CLIPModel.from_pretrained(model_id)
embedding_model.eval()

def extract_embedding(image_path: str) -> list:
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        features = embedding_model.get_image_features(**inputs)
        features = features.pooler_output
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        
    return features.squeeze().tolist()

# db 설계
DB_PATH = "/app/graph/image_quality_inspection_service/chroma_storage"
COLLECTION_NAME = "defect_images"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"} # 코사인 유사도 기반 검색
)

# 최초 1회 실행 파트
def setup_initial_database(): # 실제 서비스 시 url 변경!
    initial_defect_data = [
        {
            "id": "rgb_focus_failure",
            "path": "/app/data/rgb_focus_failure.jpg",
            "metadata": {
                "image_type": "RGB",
                "failType": "rgb_focus_failure", 
                "url": "/app/data/rgb_focus_failure.jpg"
            }
        },
        {
            "id": "rgb_hair_contamination",
            "path": "/app/data/rgb_hair_contamination.jpg",
            "metadata": {
            "image_type": "RGB",
            "failType": "rgb_hair_contamination", 
            "url": "/app/data/rgb_hair_contamination.jpg"
            }
        },
        {
            "id": "rgb_NONE",
            "path": "/app/data/rgb_NONE.jpg",
            "metadata": {
            "image_type": "RGB",
            "failType": "rgb_NONE", 
            "url": "/app/data/rgb_NONE.jpg"
            }
        },
        {
            "id": "rgb_overexposure",
            "path": "/app/data/rgb_overexposure.jpg",
            "metadata": {
            "image_type": "RGB",
            "failType": "rgb_overexposure", 
            "url": "/app/data/rgb_overexposure.jpg"
            }
        },
        {
            "id": "rgb_reflection_glare",
            "path": "/app/data/rgb_reflection_glare.jpg",
            "metadata": {
            "image_type": "RGB",
            "failType": "rgb_reflection_glare", 
            "url": "/app/data/rgb_reflection_glare.jpg"
            }
        },
        {
            "id": "rgb_surface_dust",
            "path": "/app/data/rgb_surface_dust.jpg",
            "metadata": {
            "image_type": "RGB",
            "failType": "rgb_surface_dust", 
            "url": "/app/data/rgb_surface_dust.jpg"
            }
        },
        {
            "id": "rgb_trigger_timing_failure",
            "path": "/app/data/rgb_trigger_timing_failure.jpg",
            "metadata": {
            "image_type": "RGB",
            "failType": "rgb_trigger_timing_failure", 
            "url": "/app/data/rgb_trigger_timing_failure.jpg"
            }
        },
        {
            "id": "rgb_underexposure",
            "path": "/app/data/rgb_underexposure.jpg",
            "metadata": {
            "image_type": "RGB",
            "failType": "rgb_underexposure", 
            "url": "/app/data/rgb_underexposure.jpg"
            }
        },
        {
            "id": "rgb_uneven_lighting",
            "path": "/app/data/rgb_uneven_lighting.jpg",
            "metadata": {
            "image_type": "RGB",
            "failType": "rgb_uneven_lighting", 
            "url": "/app/data/rgb_uneven_lighting.jpg"
            }
        },
        {
            "id": "ct_cell_alignment_failure",
            "path": "/app/data/ct_cell_alignment_failure.jpg",
            "metadata": {
            "image_type": "CT",
            "failType": "ct_cell_alignment_failure", 
            "url": "/app/data/ct_cell_alignment_failure.jpg"
            }
        },
        {
            "id": "ct_acquisition_motion",
            "path": "/app/data/ct_acquisition_motion.jpg",
            "metadata": {
            "image_type": "CT",
            "failType": "ct_acquisition_motion", 
            "url": "/app/data/ct_acquisition_motion.jpg"
            }
        },
        {
            "id": "ct_insufficient_projection_sampling",
            "path": "/app/data/ct_insufficient_projection_sampling.jpg",
            "metadata": {
            "image_type": "CT",
            "failType": "ct_insufficient_projection_sampling", 
            "url": "/app/data/ct_insufficient_projection_sampling.jpg"
            }
        },
        {
            "id": "ct_low_signal_noise",
            "path": "/app/data/ct_low_signal_noise.jpg",
            "metadata": {
            "image_type": "CT",
            "failType": "ct_low_signal_noise", 
            "url": "/app/data/ct_low_signal_noise.jpg"
            }
        },
        {
            "id": "ct_beam_hardening_metal_streak",
            "path": "/app/data/ct_beam_hardening_metal_streak.jpg",
            "metadata": {
            "image_type": "CT",
            "failType": "ct_beam_hardening_metal_streak", 
            "url": "/app/data/ct_beam_hardening_metal_streak.jpg"
            }
        },
        {
            "id": "ct_NONE",
            "path": "/app/data/ct_NONE.jpg",
            "metadata": {
            "image_type": "CT",
            "failType": "ct_NONE", 
            "url": "/app/data/ct_NONE.jpg"
            }
        },
    ]

    ids, embeddings, metadatas = [], [], []

    for data in initial_defect_data:
        if not os.path.exists(data["path"]):
            print(f"경고: 로컬에서 이미지를 찾을 수 없습니다 -> {data['path']}")
            continue
            
        vector = extract_embedding(data["path"])
        
        ids.append(data["id"])
        embeddings.append(vector)
        metadatas.append(data["metadata"])

    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"총 {len(ids)}건의 초기 레퍼런스 데이터 적재 완료!")

if __name__ == "__main__":
    setup_initial_database()