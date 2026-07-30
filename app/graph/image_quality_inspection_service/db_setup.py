import os
import chromadb
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 1. 임베딩 용 CLIP 모델 및 프로세서 설정
model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
embedding_model = CLIPModel.from_pretrained(model_id)
embedding_model.eval()

def extract_embedding(image_path: str) -> list:
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        features = embedding_model.get_image_features(**inputs)
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
def setup_initial_database():
    initial_defect_data = [
        {
            "id": "ref_scratch_001",
            "path": "/app/data/defects/scratch_01.jpg",
            "metadata": {
                "failType": "SCRATCH", 
                "url": "s3://my-bucket/defects/scratch_01.jpg"
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