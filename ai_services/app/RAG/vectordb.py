from qdrant_client import QdrantClient

class RecipeVectorDB:
    """
    Wrapper cho Qdrant Vector Database.
    """
    def __init__(self):
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection = "recipes"

    def search(self, vector, limit=10):
        """
        Tìm kiếm vector tương đồng (Similarity Search).
        """
        return self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=limit
        )
