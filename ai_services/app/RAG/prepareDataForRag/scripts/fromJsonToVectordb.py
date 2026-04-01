import os
import json
import uuid
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load env vars for local testing
load_dotenv()

COLLECTION_NAME = "recipes"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

def build_embedding_text(recipe):
    gia_vi_str = ", ".join(recipe.get('gia_vi', []))
    return (
        f"Món ăn: {recipe.get('ten_mon', '')}. "
        f"Mô tả: {recipe.get('mo_ta', '')}. "
        f"Nguyên liệu: {recipe.get('nguyen_lieu_search', '')}. "
        f"Gia vị: {gia_vi_str}"
    )

def run_ingestion():
    # 1. Config from Env
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    
    pg_db = os.getenv("POSTGRES_DB", "smartchef_db")
    pg_user = os.getenv("POSTGRES_USER", "admin")
    pg_pass = os.getenv("POSTGRES_PASSWORD", "admin")
    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_port = os.getenv("POSTGRES_PORT", 5432)

    # 2. Init Clients
    print(f"🔄 Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port, check_compatibility=False)
    
    print(f"🔄 Connecting to Postgres at {pg_host}:{pg_port}...")
    conn = psycopg2.connect(
        dbname=pg_db,
        user=pg_user,
        password=pg_pass,
        host=pg_host,
        port=pg_port
    )
    cursor = conn.cursor()

    # 3. Load Model
    print(f"🧬 Loading Embedding Model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 4. Prepare Collection
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"✅ Created new collection: {COLLECTION_NAME}")
    else:
        print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists. Syncing data...")

    # 5. Load Data
    # Import paths here to avoid circular imports if any
    from app.config.paths import RECIPES_JSON_PATH
    
    if not os.path.exists(RECIPES_JSON_PATH):
        print(f"❌ Error: Dataset not found at {RECIPES_JSON_PATH}")
        return

    with open(RECIPES_JSON_PATH, "r", encoding="utf-8") as f:
        recipes_data = json.load(f)

    print(f"🚀 Ingesting {len(recipes_data)} recipes into Vector DB and Postgres...")

    # 6. Create Table in Postgres
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

    # 7. Processing Loop
    for recipe in recipes_data:
        # Vector DB Ingestion
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

        # Postgres Ingestion
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
    
    conn.commit()
    cursor.close()
    conn.close()
    print("✨ Data Ingestion Completed Successfully!")

if __name__ == "__main__":
    run_ingestion()
