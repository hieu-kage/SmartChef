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
        """
        text = "query: " + ", ".join(ingredients)
        return self.model.encode(text).tolist()
