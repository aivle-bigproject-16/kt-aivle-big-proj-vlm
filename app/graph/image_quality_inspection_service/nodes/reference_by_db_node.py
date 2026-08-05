from app.graph.image_quality_inspection_service.state import QualityState
from app.graph.image_quality_inspection_service.db_setup import collection, extract_embedding

def search_similar_defects(target_image_path: str, top_k: int = 1, image_type: str = "") -> dict:
    target_embedding = extract_embedding(target_image_path)
    
    result = collection.query(
        query_embeddings=[target_embedding],
        n_results=top_k,
        include=["metadatas"],
        where={"image_type": image_type}
    )
    
    return result

async def reference_by_db_node(state: QualityState) -> dict:
    images = state.get("images", [])
    image_type = state.get("imageType", "")
    reference_cases = []
    
    for img in images:
        result = search_similar_defects(img["imageUrl"], top_k=1, image_type= image_type)

        if result and result.get("metadatas") and result["metadatas"][0]:
            ref_meta = result["metadatas"][0][0]
            
            reference_cases.append({
                "target_imageId": img["imageId"],
                "ref_imageUrl": ref_meta.get("url", "UNKNOWN"),
                "ref_failType": ref_meta.get("failType", "UNKNOWN")
            })
            
    return {"reference_cases": reference_cases}