"""
Module: SmartChef API Routes
==========================

Định nghĩa các API endpoint cho dịch vụ AI.

Nhiệm vụ:
- Tiếp nhận yêu cầu (request) từ phía client.
- Gọi xuống lớp Service để xử lý nghiệp vụ (Vision, RAG, LLM).
- Trả về kết quả (response) chuẩn hóa dưới dạng JSON.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from .service import SmartChefService
import os

router = APIRouter()


smart_chef = SmartChefService()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@router.post("/predict", summary="Nhận diện nguyên liệu & Gợi ý (Start Session)")
async def predict(
    file: UploadFile = File(...),
    session_id: str = Form("default")
):
    """
    Endpoint khởi tạo phiên làm việc (Session Start).
    
    Quy trình:
    1. Tiếp nhận ảnh nguyên liệu từ người dùng.
    2. Sử dụng YOLO để nhận diện danh sách nguyên liệu.
    3. Gợi ý công thức nấu ăn dựa trên nguyên liệu (RAG + LLM).
    4. Trả về kết quả và khởi tạo context cho session.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Read file content directly into memory
        file_bytes = await file.read()
            
        result = smart_chef.suggest_recipes(file_bytes, session_id)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", summary="Hội thoại với AI (Follow-up)")
async def chat(request: ChatRequest):
    """
    Endpoint xử lý hội thoại (Follow-up Chat).
    
    Quy trình:
    1. Tiếp nhận câu hỏi hoặc phản hồi từ người dùng.
    2. Truy xuất lịch sử hội thoại dựa trên `session_id`.
    3. Sử dụng LLM để sinh câu trả lời phù hợp ngữ cảnh.
    """
    try:
        response = smart_chef.chat(request.session_id, request.message)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
