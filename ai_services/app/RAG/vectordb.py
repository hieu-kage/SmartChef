from qdrant_client import QdrantClient
import os
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
class RecipeVectorDB:
    """
    Wrapper cho Qdrant Vector Database.
    """
    def __init__(self):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
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
