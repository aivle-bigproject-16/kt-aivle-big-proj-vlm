from app.graph.image_quality_inspection_service.state import QualityState

async def reference_by_db_node(state: QualityState) -> dict:
    images = state.get("images", [])
    reference_cases = []
    
    for img in images:
        # 벡터 DB와 유사도 검사
        results = "" #search_similar_defects(img["imageUrl"], top_k=1)

        if results and results.get("metadatas") and results["metadatas"][0]:
            ref_meta = results["metadatas"][0][0]
            ref_uri = results["uris"][0][0]
            
            reference_cases.append({
                "target_imageId": img["imageId"],
                "ref_imageUrl": ref_uri,
                "ref_failType": ref_meta.get("failType", "UNKNOWN"),
                "ref_description": ref_meta.get("description", "")
            })
            
    return {"reference_cases": reference_cases}