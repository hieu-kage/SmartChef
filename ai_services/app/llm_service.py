"""
Module: LLM Service
=================

Cung cấp dịch vụ Xử lý Ngôn ngữ Tự nhiên (NLP) thông qua Google Gemini và LangChain.

Chức năng:
- Quản lý ngữ cảnh hội thoại (Context Management) theo session.
- Sinh nội dung tư vấn món ăn dựa trên nguyên liệu và công thức.
- Xử lý các tác vụ Chat General.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

class LLMService:
    """
    Service wrapper cho Google Gemini API và LangChain.
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            print("Warning: GOOGLE_API_KEY not found. LLM service will not work.")
        
        self.model_name = os.getenv("MODEL_NAME", "gemini-1.5-flash")
        
        if self.api_key:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                google_api_key=self.api_key
            )
            
            self.suggestion_prompt = ChatPromptTemplate.from_messages(messages=[
                ("system", """
Bạn là SmartChef - Chuyên gia ẩm thực thông minh.
Người dùng đang có các nguyên liệu sau: **{ingredients}**

Tôi vừa tìm thấy các công thức sau trong cơ sở dữ liệu phù hợp với nguyên liệu của bạn:
{recipe_context}

Nhiệm vụ của bạn:
1. Xác nhận nguyên liệu người dùng có: Liệt kê lại 3-5 nguyên liệu quan trọng nhất từ input của người dùng (để chắc chắn bạn không bị nhầm lẫn).
2. CHỌN 1 món ăn tốt nhất trong danh sách trên để hướng dẫn. 
   **QUAN TRỌNG:** So sánh Nguyên Liệu Chính của món ăn với danh sách bạn vừa liệt kê ở bước 1.
   - Nếu món ăn yêu cầu "Thịt bò" mà người dùng chỉ có "Thịt heo" -> LOẠI NGAY (dù điểm match cao).

3. Trả lời theo cấu trúc sau:
   - "Dựa trên nguyên liệu của bạn (gồm ...), tôi đề xuất món: [Tên Món Chọn]"
   - Giải thích ngắn gọn tại sao chọn món này.
   - Hướng dẫn chi tiết cách làm (Dựa trên thông tin "Cách làm" đã cung cấp trong context).
   - Mẹo nhỏ để món ăn ngon hơn.

Lưu ý:
- Chỉ sáng tạo món mới NẾU VÀ CHỈ NẾU danh sách trên hoàn toàn không phù hợp (ví dụ sai lệch nguyên liệu chính).
- Tuyệt đối không nói "Dựa trên danh sách bạn cung cấp", hãy nói "Hệ thống tìm thấy...".
"""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"), 
            ])

            self.chat_prompt = ChatPromptTemplate.from_messages(messages=[
                ("system", "Bạn là SmartChef. Hãy trả lời câu hỏi của người dùng dựa trên ngữ cảnh các món ăn và nguyên liệu đã thảo luận trước đó."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])
            
            self.suggestion_chain = RunnableWithMessageHistory(
                runnable=self.suggestion_prompt | self.llm | StrOutputParser(),
                get_session_history=get_session_history,
                input_messages_key="question",
                history_messages_key="history",
            )
            
            self.chat_chain = RunnableWithMessageHistory(
                runnable=self.chat_prompt | self.llm | StrOutputParser(),
                get_session_history=get_session_history,
                input_messages_key="question",
                history_messages_key="history",
            )

    def generate_suggestion(self, session_id: str, ingredients: list[str], recipes: list[dict]) -> str:
        """
        Sinh nội dung tư vấn món ăn từ LLM.

        Args:
            session_id (str): ID phiên làm việc.
            ingredients (list[str]): Danh sách nguyên liệu đầu vào.
            recipes (list[dict]): Danh sách công thức đã tìm được từ RAG.

        Returns:
            str: Nội dung tư vấn từ AI.
        """
        if not self.api_key: return "LLM not configured."

        ingredient_str = ", ".join(ingredients)
        recipe_context = ""
        for i, r in enumerate(recipes):
            recipe_context += f"{i+1}. Tên món: {r['ten_mon']} (Độ khớp: {r['match_score']:.2f})\n"
            recipe_context += f"   - Mô tả: {r.get('mo_ta', '')}\n"
            
            nguyen_lieu = ", ".join(r.get('nguyen_lieu_chi_tiet', [])) or "Không có thông tin"
            recipe_context += f"   - Nguyên liệu chi tiết: {nguyen_lieu}\n"
            
            gia_vi = ", ".join(r.get('gia_vi', [])) or "Không có thông tin"
            recipe_context += f"   - Gia vị: {gia_vi}\n"
            
            steps = r.get('cach_lam', [])
            if steps:
                cach_lam_str = "\n".join([f"     + {step}" for step in steps])
            else:
                cach_lam_str = "     + Không có thông tin"
            recipe_context += f"   - Cách làm:\n{cach_lam_str}\n\n"

        try:
            user_msg = "Hãy gợi ý món ăn cho tôi dựa trên các nguyên liệu này."
            
            response = self.suggestion_chain.invoke(
                input={
                    "ingredients": ingredient_str,
                    "recipe_context": recipe_context,
                    "question": user_msg
                },
                config={"configurable": {"session_id": session_id}}
            )
            return response
        except Exception as e:
            return f"Error generating suggestion: {str(e)}"

    def chat(self, session_id: str, message: str) -> str:
        """
        Trả lời câu hỏi của người dùng dựa trên lịch sử chat.
        """
        if not self.api_key: return "LLM not configured."
        
        try:
            response = self.chat_chain.invoke(
                {"question": message},
                config={"configurable": {"session_id": session_id}}
            )
            return response
        except Exception as e:
            return f"Error replying to chat: {str(e)}"
