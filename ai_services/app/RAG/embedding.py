from sentence_transformers import SentenceTransformer

class EmbeddingService:
    """
    Service tạo vector embedding cho văn bản.
    Sử dụng model: intfloat/multilingual-e5-base (768-dims).
    """
    def __init__(self):
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-base"
        )

    def embed_ingredients(self, ingredients: list[str]) -> list[float]:
        """
        Chuyển danh sách nguyên liệu thành vector 768-dims.
        Tối ưu hóa chuỗi truy vấn (query) để khai thác kiến thức ẩm thực của model E5.
        """
        if not ingredients:
            return [0.0] * 768
            
        ingredients_str = ", ".join(ingredients)
        # Sử dụng cấu trúc câu hoàn chỉnh để Model "hiểu" ngữ cảnh tìm kiếm món ăn
        text = f"query: Tìm các món ăn Việt Nam ngon và phổ biến nhất được chế biến từ các nguyên liệu sau: {ingredients_str}."
        return self.model.encode(text).tolist()
