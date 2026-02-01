"""
Module: SmartChef Orchestration Service
=====================================

Lớp điều phối trung tâm (Orchestrator) tích hợp các dịch vụ AI thành quy trình thống nhất.

Quy trình `suggest_recipes`:
1. Vision Layer: Phân tích ảnh, trích xuất nguyên liệu (Yolo).
2. Retrieval Layer: Tìm kiếm công thức dựa trên nguyên liệu (RAG).
3. Generation Layer: Tổng hợp và tư vấn (LLM).

Quy trình `chat`:
- Quản lý hội thoại ngữ cảnh lỏng (Contextual Conversation) thông qua LLM Service.
"""

from .Vison.yolo_service import YoloIngredientService
from .RAG.rag_service import RecipeRAGService
from .llm_service import LLMService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartChefService:
    """
    Lớp điều phối chính cho hệ thống SmartChef AI.
    
    Attributes:
        yolo_service (YoloIngredientService): Service nhận diện nguyên liệu.
        rag_service (RecipeRAGService): Service truy vấn công thức.
        llm_service (LLMService): Service xử lý ngôn ngữ và sinh nội dung.
    """
    def __init__(self):
        logger.info("Initializing SmartChef AI Services...")
        try:
            self.yolo_service = YoloIngredientService()
            logger.info("Yolo Service initialized.")
        except Exception as e:
            logger.error(f"Failed to init Yolo Service: {e}")
            self.yolo_service = None

        try:
            self.rag_service = RecipeRAGService()
            logger.info("RAG Service initialized.")
        except Exception as e:
            logger.error(f"Failed to init RAG Service: {e}")
            self.rag_service = None

        try:
            self.llm_service = LLMService()
            logger.info("LLM Service initialized.")
        except Exception as e:
            logger.error(f"Failed to init LLM Service: {e}")
            self.llm_service = None
            
        logger.info("SmartChef AI Services initialized.")

    def suggest_recipes(self, image_bytes: bytes, session_id: str = "default") -> dict:
        """
        Thực hiện quy trình đầy đủ: Nhận diện -> Tìm kiếm -> Tư vấn.

        Args:
            image_bytes (bytes): Dữ liệu ảnh dạng byte.
            session_id (str): ID phiên làm việc của người dùng.

        Returns:
            dict: Kết quả bao gồm nguyên liệu, danh sách công thức và lời khuyên từ AI.
        """
        result = {
            "detected_ingredients": [],
            "recipes": [],
            "llm_suggestion": ""
        }

        if self.yolo_service:
            try:
                ingredients = self.yolo_service.detect_ingredients(image_bytes)
                result["detected_ingredients"] = ingredients
                logger.info(f"Detected ingredients: {ingredients}")
            except Exception as e:
                logger.error(f"Error in Yolo detection: {e}")
                return {"error": f"Yolo Error: {str(e)}"}
        else:
             logger.warning("Yolo Service not available.")
             return {"error": "Yolo Service not available"}

        if not ingredients:
            result["llm_suggestion"] = "Không tìm thấy nguyên liệu nào trong ảnh."
            return result

        if self.rag_service:
            try:
                recipes = self.rag_service.retrieve(ingredients)
                result["recipes"] = recipes
                logger.info(f"Retrieved {len(recipes)} recipes.")
            except Exception as e:
                logger.error(f"Error in RAG retrieval: {e}")
                
        if self.llm_service:
            try:
                suggestion = self.llm_service.generate_suggestion(
                    session_id,
                    result["detected_ingredients"],
                    result["recipes"]
                )
                result["llm_suggestion"] = suggestion
                logger.info("LLM suggestion generated.")
            except Exception as e:
                logger.error(f"Error in LLM generation: {e}")
                result["llm_suggestion"] = "Lỗi khi tạo gợi ý từ AI."

        return result

    def chat(self, session_id: str, message: str) -> dict:
        """
        Xử lý hội thoại tiếp diễn với người dùng.

        Args:
            session_id (str): ID phiên làm việc hiện tại.
            message (str): Câu hỏi hoặc tin nhắn từ người dùng.

        Returns:
            dict: Câu trả lời từ AI hoặc thông báo lỗi.
        """
        if self.llm_service:
            try:
                reply = self.llm_service.chat(session_id, message)
                return {"reply": reply}
            except Exception as e:
                logger.error(f"Error in Chat: {e}")
                return {"error": str(e)}
        return {"error": "LLM Service not available"}