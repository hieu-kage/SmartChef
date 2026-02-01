from .embedding import EmbeddingService
from .vectordb import RecipeVectorDB

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
        
        union = recipe_set | query_set
        
        return len(intersection) / len(union)

    def retrieve(self, ingredients: list[str], top_k=5):
        """
        Tìm kiếm công thức phù hợp dựa trên danh sách nguyên liệu.
        Kết hợp Semantic Search (Vector) và Keyword Matching.
        """
        query_vector = self.embedding.embed_ingredients(ingredients)
        hits = self.vectordb.search(query_vector, limit=top_k * 2)

        results = []
        for hit in hits:
            payload = hit.payload
            recipe_ingredients_str = payload.get("nguyen_lieu_search", "")
            recipe_ingredients = [i.strip().lower() for i in recipe_ingredients_str.split(",")]
            
            match_score = self.ingredient_match_score(
                recipe_ingredients,
                ingredients
            )

            if match_score >= 0.2:
                results.append({
                    "id": payload["id"],
                    "ten_mon": payload["ten_mon"],
                    "match_score": match_score,
                    "semantic_score": hit.score
                })

        results.sort(
            key=lambda x: (x["match_score"], x["semantic_score"]),
            reverse=True
        )

        return results[:top_k]
