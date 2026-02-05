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


    def ingredient_match_score(self, recipe_ingredients, query_ingredients):
        """
        Tính điểm độ khớp nguyên liệu (Intersection over Query).
        """
        recipe_set = set(recipe_ingredients)
        query_set = set(query_ingredients)

        if not query_set or not recipe_set:
            return 0.0

        intersection = recipe_set & query_set
        return len(intersection) / len(recipe_set)

    def retrieve(self, ingredients: list[str], top_k=5):
        """
        Tìm kiếm công thức phù hợp dựa trên danh sách nguyên liệu.
        Kết hợp Semantic Search (Vector) và Keyword Matching.
        """
        query_vector = self.embedding.embed_ingredients(ingredients)
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
            
            recipe_ingredients_str = payload.get("nguyen_lieu_search", "")
            recipe_ingredients = [i.strip().lower() for i in recipe_ingredients_str.split(",")]
            
            match_score = self.ingredient_match_score(
                recipe_ingredients,
                ingredients
            )

            print(f"--- Đánh giá món: {payload.get('ten_mon')} ---")
            print(f"Điểm Match: {match_score:.2f}")

            if match_score >= 0.2:
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
