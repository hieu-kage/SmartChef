from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import psycopg2
import json
import uuid

COLLECTION_NAME = "recipes"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
model = SentenceTransformer(EMBEDDING_MODEL)
from ai_services.app.config.paths import RECIPES_JSON_PATH
qdrant = QdrantClient(
    host="localhost",
    port=6333
)

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )
conn = psycopg2.connect(
    dbname="smartchef_db",
    user="admin",
    password="admin",
    host="localhost",
    port=5432
)

cursor = conn.cursor()

def build_embedding_text(recipe):
    return (
        f"Món ăn: {recipe['ten_mon']}. "
        f"Mô tả: {recipe['mo_ta']}. "
        f"Nguyên liệu: {recipe['nguyen_lieu_search']}.")

def insert_recipe(recipe):
    text = build_embedding_text(recipe)

    vector = model.encode(f"passage: {text}").tolist()
    point_id = str(uuid.uuid4())
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "recipe_id": recipe["id"],
                    "ten_mon": recipe["ten_mon"],
                    "nguyen_lieu_search": recipe["nguyen_lieu_search"]
                }
            )
        ]
    )

    cursor.execute(
        """
        INSERT INTO recipes (
            id, ten_mon, mo_ta, nguyen_lieu_search,
            nguyen_lieu_chi_tiet, cach_lam, thoi_gian_nau
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
        """,
        (
            recipe["id"],
            recipe["ten_mon"],
            recipe["mo_ta"],
            recipe["nguyen_lieu_search"],
            json.dumps(recipe["nguyen_lieu_chi_tiet"]),
            json.dumps(recipe["cach_lam"]),
            recipe["thoi_gian_nau"]
        )
    )
with open(RECIPES_JSON_PATH, "r", encoding="utf-8") as f:
    recipes_data = json.load(f)

for recipe in recipes_data:
    insert_recipe(recipe)

conn.commit()
print("Ingest xong dữ liệu")




