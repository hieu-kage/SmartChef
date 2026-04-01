"""
Module: SmartChef AI Integration Service
======================================

Điểm truy cập chính (Entry Point) của ứng dụng SmartChef AI.

Nhiệm vụ:
- Khởi tạo ứng dụng FastAPI.
- Cấu hình CORS Middleware.
- Tích hợp các Router API.
- Cấu hình Uvicorn server để triển khai ứng dụng.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import router as api_router
import uvicorn
import os
import sys
from contextlib import asynccontextmanager
from qdrant_client import QdrantClient
from .RAG.prepareDataForRag.scripts.fromJsonToVectordb import run_ingestion

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("🚀 SmartChef AI Service is starting up...")
    
    # 🔍 Kiểm tra Dữ liệu (Auto-Ingestion)
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    
    try:
        qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        collection_name = "recipes"
        
        # Nếu collection chưa tồn tại hoặc rỗng, tiến hành nạp dữ liệu
        should_ingest = False
        if not qdrant.collection_exists(collection_name):
            print(f"⚠️ Collection '{collection_name}' không tồn tại. Đang khởi tạo dữ liệu...")
            should_ingest = True
        else:
            count = qdrant.get_collection(collection_name).points_count
            if count == 0:
                print(f"⚠️ Collection '{collection_name}' đang trống. Đang nạp dữ liệu...")
                should_ingest = True
        
        if should_ingest:
            run_ingestion()
            print("✅ Tự động nạp dữ liệu hoàn tất!")
        else:
            print(f"✅ Dữ liệu '{collection_name}' đã sẵn sàng ({qdrant.get_collection(collection_name).points_count} món ăn).")
            
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra dữ liệu khởi động: {e}")
        print("⚠️ Ứng dụng sẽ tiếp tục chạy, nhưng RAG có thể không hoạt động!")

    yield
    # --- SHUTDOWN ---
    print("💤 SmartChef AI Service is shutting down...")

app = FastAPI(
    title="SmartChef AI Services",
    description="API Service tích hợp Vision, RAG và LLM cho SmartChef System",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "SmartChef AI Service is Ready!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
