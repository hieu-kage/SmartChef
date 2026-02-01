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

app = FastAPI(
    title="SmartChef AI Services",
    description="API Service tích hợp Vision, RAG và LLM cho SmartChef System",
    version="2.0"
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
