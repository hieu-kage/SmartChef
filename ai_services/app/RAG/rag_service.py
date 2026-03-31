from .embedding import EmbeddingService
from .vectordb import RecipeVectorDB
from app import db

class RecipeRAGService:
    """
    Service quản lý việc truy xuất công thức nấu ăn (Retrieval-Augmented Generation).
    """
    def __init__(self):
        self.embedding = EmbeddingService()
        self.vectordb = RecipeVectorDB()

    def ingredient_match_score(self, required_ingredients, available_ingredients):
        """
        Khớp chuỗi linh hoạt: So sánh đồ YÊU CẦU (required) với đồ SẴN CÓ (available).
        """
        if not available_ingredients or not required_ingredients:
            return 0.0

        avail_norm = [a.lower().strip() for a in available_ingredients]
        req_norm = [r.lower().strip() for r in required_ingredients]

        matched_count = 0
        for req in req_norm:
            is_matched = False
            for avail in avail_norm:
                if req in avail or avail in req:
                    is_matched = True
                    break

            if is_matched:
                matched_count += 1

        return matched_count / len(req_norm)

    def retrieve(self, available_ingredients: list[str], top_k=5):
        """
        Tìm kiếm công thức phù hợp dựa trên danh sách nguyên liệu.
        Kết hợp Semantic Search (Vector) và Keyword Matching.
        """
        query_vector = self.embedding.embed_ingredients(available_ingredients)
        hits = self.vectordb.search(query_vector, limit=top_k * 2)
        print(f"Vector Hits: {len(hits)}")

        if not hits:
            return []

        hit_ids = [hit.payload["recipe_id"] for hit in hits]
        db_recipes = db.get_recipes_by_ids(hit_ids)
        
        results = []
        for hit in hits:
            payload = hit.payload
            r_id = payload["recipe_id"]
            
            recipe_detail = db_recipes.get(r_id, {})

            required_ingredients_str = payload.get("nguyen_lieu_search", "")
            required_ingredients = [i.strip().lower() for i in required_ingredients_str.split(",")]
            
            match_score = self.ingredient_match_score(
                required_ingredients,
                available_ingredients
            )

            print(f"--- Đánh giá món: {payload.get('ten_mon')} ---")
            print(f"Điểm Match: {match_score:.2f}")

            if match_score >= 0.7:
                results.append({
                    "id": r_id,
                    "ten_mon": payload["ten_mon"],
                    "match_score": match_score,
                    "semantic_score": hit.score,
                    "nguyen_lieu_chi_tiet": recipe_detail.get("nguyen_lieu_chi_tiet", []),
                    "gia_vi": recipe_detail.get("gia_vi", []),
                    "cach_lam": recipe_detail.get("cach_lam", []),
                    "mo_ta": recipe_detail.get("mo_ta", "")
                })

        results.sort(
            key=lambda x: (x["match_score"], x["semantic_score"]),
            reverse=True
        )

        return results[:top_k]




