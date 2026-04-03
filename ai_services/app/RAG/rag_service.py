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
    # Nhóm các nguyên liệu "nhiễu" (Gia vị/Hành tỏi) để giảm ưu tiên
    NOISY_KEYWORDS = ["tỏi", "hành", "tiêu", "muối", "đường", "vị", "dầu", "nước mắm"]

    def ingredient_match_score(self, required_ingredients, available_ingredients):
        """
        Tính điểm khớp có trọng số:
        - Khớp nguyên liệu chính (Thịt/Cá): x3.0 điểm.
        - Khớp rau củ/tinh bột: x1.5 điểm.
        - Khớp gia vị: x0.5 điểm.
        """
        if not available_ingredients or not required_ingredients:
            return 0.0

        avail_norm = [a.lower().strip() for a in available_ingredients]
        req_norm = [r.lower().strip() for r in required_ingredients]

        total_weight = 0.0
        matched_weight = 0.0

        for req in req_norm:
            # Xác định trọng số của nguyên liệu này
            weight = 1.0 # Mặc định cho rau củ
            if any(k in req for k in self.PRIMARY_KEYWORDS):
                weight = 3.0
            elif any(k in req for k in self.NOISY_KEYWORDS):
                weight = 0.3 # Giảm hẳn trọng số gia vị

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
        # 1. Vector Search (Lấy mẫu rộng hơn để lọc - 30 hits)
        query_vector = self.embedding.embed_ingredients(available_ingredients)
        hits = self.vectordb.search(query_vector, limit=30)
        
        if not hits:
            return []

        # 2. Lấy chi tiết từ Postgres
        hit_ids = [hit.payload["recipe_id"] for hit in hits]
        db_recipes = db.get_recipes_by_ids(hit_ids)
        
        candidates = []
        for hit in hits:
            payload = hit.payload
            r_id = payload["recipe_id"]
            recipe_detail = db_recipes.get(r_id, {})

            required_ingredients_str = payload.get("nguyen_lieu_search", "")
            required_ingredients = [i.strip().lower() for i in required_ingredients_str.split(",")]
            
            # Tính điểm khớp có trọng số
            match_score = self.ingredient_match_score(
                required_ingredients,
                available_ingredients
            )
            
            # Tính điểm tổng hợp (Weighted Logic)
            # - Tăng giá trị của match_score vì nó thực tế hơn
            combined_score = (match_score * 0.7) + (hit.score * 0.3)

            if match_score >= 0.5: # Giảm ngưỡng để lấy được nhiều món tiềm năng hơn
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




