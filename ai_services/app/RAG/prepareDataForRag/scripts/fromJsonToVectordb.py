from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import psycopg2
import json




import uuid

COLLECTION_NAME = "recipes"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
model = SentenceTransformer(EMBEDDING_MODEL)
from app.config.paths import RECIPES_JSON_PATH
qdrant = QdrantClient(
    host="localhost",
    port=6333,
    check_compatibility=False
)

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )
    print(f"Đã tạo collection mới: {COLLECTION_NAME}")
else:
    print(f" Collection '{COLLECTION_NAME}' đã tồn tại -> Chế độ Incremental Upsert.")

conn = psycopg2.connect(
    dbname="smartchef_db",
    user="admin",
    password="admin",
    host="localhost",
    port=5432
)

cursor = conn.cursor()

def build_embedding_text(recipe):
    gia_vi_str = ", ".join(recipe.get('gia_vi', []))
    return (
        f"Món ăn: {recipe.get('ten_mon', '')}. "
        f"Mô tả: {recipe.get('mo_ta', '')}. "
        f"Nguyên liệu: {recipe.get('nguyen_lieu_search', '')}. "
        f"Gia vị: {gia_vi_str}"
    )

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
                    "recipe_id": recipe.get("id"),
                    "ten_mon": recipe.get("ten_mon"),
                    "nguyen_lieu_search": recipe.get("nguyen_lieu_search"),
                    "gia_vi": recipe.get("gia_vi", [])
                }
            )
        ]
    )

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recipes (
            id TEXT PRIMARY KEY,
            ten_mon TEXT,
            mo_ta TEXT,
            nguyen_lieu_search TEXT,
            nguyen_lieu_chi_tiet JSONB,
            cach_lam JSONB,
            thoi_gian_nau TEXT,
            gia_vi JSONB
        );
    """)
    
    conn.commit()

    cursor.execute(
        """
        INSERT INTO recipes (
            id, ten_mon, mo_ta, nguyen_lieu_search,
            nguyen_lieu_chi_tiet, cach_lam, thoi_gian_nau, gia_vi
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            ten_mon = EXCLUDED.ten_mon,
            mo_ta = EXCLUDED.mo_ta,
            nguyen_lieu_search = EXCLUDED.nguyen_lieu_search,
            nguyen_lieu_chi_tiet = EXCLUDED.nguyen_lieu_chi_tiet,
            cach_lam = EXCLUDED.cach_lam,
            thoi_gian_nau = EXCLUDED.thoi_gian_nau,
            gia_vi = EXCLUDED.gia_vi
        """,
        (
            recipe.get("id"),
            recipe.get("ten_mon"),
            recipe.get("mo_ta"),
            recipe.get("nguyen_lieu_search"),
            json.dumps(recipe.get("nguyen_lieu_chi_tiet", [])),
            json.dumps(recipe.get("cach_lam", [])),
            recipe.get("thoi_gian_nau"),
            json.dumps(recipe.get("gia_vi", []))
        )
    )

with open(RECIPES_JSON_PATH, "r", encoding="utf-8") as f:
    recipes_data = json.load(f)

print(f"Đang reset collection '{COLLECTION_NAME}' và nạp {len(recipes_data)} món ăn...")

for recipe in recipes_data:
    insert_recipe(recipe)

conn.commit()
print(" Ingest xong dữ liệu! (DB Cleaned & Updated)")




