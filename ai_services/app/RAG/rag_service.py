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

    # Nhóm các nguyên liệu chính (Protein/Đặc trưng) để ưu tiên
    PRIMARY_KEYWORDS = [
        "thịt", "cá", "tôm", "cua", "gà", "bò", "lợn", "heo", "vịt", "trứng", 
        "đậu hũ", "sườn", "mực", "lươn", "ốc", "ếch", "giò", "chả", "xúc xích"
    ]
    NOISY_KEYWORDS = ["tỏi", "hành", "tiêu", "muối", "đường", "vị", "dầu", "nước mắm"]

    # Bộ từ điển đồng nghĩa để chuẩn hóa nguyên liệu
    SYNONYMS = {
        "lợn": "heo",
        "ngô": "bắp",
        "tàu hũ": "đậu phụ",
        "đậu hũ": "đậu phụ",
        "đỗ": "đậu",
        "quả": "trái",
        " thơm": " dứa",
        "khóm": " dứa",
        "mì chính": "bột ngọt"
    }

    def normalize(self, text: str) -> str:
        """Chuẩn hóa từ ngữ vùng miền và dọn dẹp chuỗi (Ví dụ: Lợn -> Heo)"""
        text = text.lower().strip()
        for k, v in self.SYNONYMS.items():
            text = text.replace(k, v)
        return text

    def ingredient_match_score(self, required_ingredients, available_ingredients):
        """
        Tính điểm khớp có trọng số:
        - Khớp nguyên liệu chính (Thịt/Cá): x3.0 điểm.
        - Khớp rau củ/tinh bột: x1.5 điểm.
        - Khớp gia vị: x0.5 điểm.
        """
        if not available_ingredients or not required_ingredients:
            return 0.0

        avail_norm = [self.normalize(a) for a in available_ingredients]
        req_norm = [self.normalize(r) for r in required_ingredients]

        total_weight = 0.0
        matched_weight = 0.0

        for req in req_norm:
            weight = 1.0 
            if any(k in req for k in self.PRIMARY_KEYWORDS):
                weight = 3.0
            elif any(k in req for k in self.NOISY_KEYWORDS):
                weight = 0.3 

            total_weight += weight
            
            is_matched = False
            for avail in avail_norm:
                if req in avail or avail in req:
                    is_matched = True
                    break

            if is_matched:
                matched_weight += weight

        return matched_weight / total_weight if total_weight > 0 else 0.0

    def retrieve(self, available_ingredients: list[str], top_k=5):
        """
        Tìm kiếm công thức phù hợp dựa trên danh sách nguyên liệu.
        Kết hợp Semantic Search (Vector) và Weighted Keyword Matching.
        """
        query_vector = self.embedding.embed_ingredients(available_ingredients)
        hits = self.vectordb.search(query_vector, limit=30)
        
        if not hits:
            return []

        hit_ids = [hit.payload["recipe_id"] for hit in hits]
        db_recipes = db.get_recipes_by_ids(hit_ids)
        
        candidates = []
        for hit in hits:
            payload = hit.payload
            r_id = payload["recipe_id"]
            recipe_detail = db_recipes.get(r_id, {})

            required_ingredients_str = payload.get("nguyen_lieu_search", "")
            required_ingredients = [i.strip().lower() for i in required_ingredients_str.split(",")]
            print(f"DEBUG: Required ingredients: {required_ingredients}")
            print(f"DEBUG: Available ingredients: {available_ingredients}")
            match_score = self.ingredient_match_score(
                required_ingredients,
                available_ingredients
            )
            

            combined_score = (match_score * 0.5) + (hit.score * 0.5)

            if combined_score >= 0.4: 
                candidates.append({
                    "id": r_id,
                    "ten_mon": payload["ten_mon"],
                    "match_score": match_score,
                    "combined_score": combined_score,
                    "semantic_score": hit.score,
                    "nguyen_lieu_chi_tiet": recipe_detail.get("nguyen_lieu_chi_tiet", []),
                    "gia_vi": recipe_detail.get("gia_vi", []),
                    "cach_lam": recipe_detail.get("cach_lam", []),
                    "mo_ta": recipe_detail.get("mo_ta", "")
                })

        # 3. Sắp xếp lại danh sách dựa trên điểm tổng hợp
        candidates.sort(
            key=lambda x: x["combined_score"],
            reverse=True
        )
        return candidates[:top_k]




