import os
import psycopg
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres import PostgresChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

class LLMService:
    """
    Service wrapper cho Google Gemini API và LangChain.
    Đã tích hợp: Postgres Database, Sliding Window Memory & Summarization.
    """
    def __init__(self):
        # 1. Cấu hình môi trường & Database
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.db_url = os.getenv("DATABASE_URL", "postgresql://admin:admin@localhost:5432/smartchef_db")
        self.model_name = os.getenv("MODEL_NAME", "gemini-1.5-flash")
        self.sync_connection = psycopg.connect(self.db_url)
        if not self.api_key:
            print("Warning: GOOGLE_API_KEY not found. LLM service will not work.")
            return

        # 2. Khởi tạo Model
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.7, # 0.1
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            google_api_key=self.api_key
        )
        
        # 3. Định nghĩa Prompts & Chains
        self._init_prompts()
        self._init_chains()

    def _init_prompts(self):
        """Khởi tạo toàn bộ khuôn mẫu Prompt"""
        
        # Prompt 1: Dùng cho lần đầu tiên user up ảnh (RAG)
        self.suggestion_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là SmartChef - Chuyên gia ẩm thực thông minh.
            
            NGUYÊN LIỆU NGƯỜI DÙNG CÓ: **{ingredients}**
            
            DANH SÁCH CÔNG THỨC TRONG DATABASE (RAG Context):
            {recipe_context}
            
            NHIỆM VỤ CỦA BẠN:
            1. Xác nhận các nguyên liệu mà người dùng đang có.
            2. BẮT BUỘC CHỈ ĐƯỢC CHỌN 1 món ăn từ "DANH SÁCH CÔNG THỨC TRONG DATABASE" ở trên. Tuyệt đối không tự ý gợi ý món ăn nằm ngoài danh sách này.
            3. So sánh nguyên liệu người dùng có với công thức đã chọn để giải thích lý do tại sao món này là phù hợp nhất (dù có thể thiếu một vài nguyên liệu phụ).
            4. Hướng dẫn cách làm chi tiết DỰA TRÊN nội dung công thức trong Database.
            5. Nếu danh sách trong Database không có món nào liên quan đến nguyên liệu của người dùng, hãy nói: "Tôi chưa tìm thấy món ăn nào hoàn toàn phù hợp trong thư viện hiện tại, nhưng dựa trên nguyên liệu của bạn, tôi khuyên bạn có thể thử tìm thêm..."
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"), 
        ])

        # Prompt 2: Dùng cho hội thoại hàng ngày (Có tiêm Summary)
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là SmartChef - Trợ lý đầu bếp thông minh.
            TÓM TẮT HỘI THOẠI QUÁ KHỨ: {summary}
            
            BẠN BẮT BUỘC PHẢI TUÂN THỦ CÁC QUY TẮC SAU:
            1. CHỈ trả lời các câu hỏi liên quan đến ẩm thực, nấu ăn, công thức, nguyên liệu và dinh dưỡng.
            2. NẾU người dùng hỏi về các chủ đề KHÁC (lập trình, toán học, chính trị, thời tiết...), BẮT BUỘC từ chối khéo léo: "Xin lỗi, tôi là đầu bếp SmartChef nên chỉ có thể giúp bạn các vấn đề về nấu ăn thôi nhé!". Tuyệt đối không trả lời nội dung lạc đề.
            3. Trả lời DỰA TRÊN ngữ cảnh và lịch sử. Nếu không biết hoặc không có thông tin trong dữ liệu, hãy nói "Tôi chưa có thông tin về món này, bạn có thể thử nguyên liệu khác không?". Không tự bịa ra công thức.
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # Prompt 3: Dùng cho tác vụ ngầm - Tóm tắt hội thoại
        self.sum_prompt = ChatPromptTemplate.from_messages([
            ("system", "Hãy tóm tắt hội thoại sau cực kỳ ngắn gọn. BẮT BUỘC giữ lại danh sách nguyên liệu và món ăn đang được nhắc tới."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Tóm tắt nội dung trên.")
        ])

    def _init_chains(self):
        """Đóng gói các chuỗi xử lý"""
        # Chain tóm tắt (Chạy ngầm, không cần lưu history)
        self.summary_chain = self.sum_prompt | self.llm | StrOutputParser()

        # Chain Suggestion (Có lưu history)
        self.suggestion_chain = RunnableWithMessageHistory(
            runnable=self.suggestion_prompt | self.llm | StrOutputParser(),
            get_session_history=self._get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        
        # Chain Chat (Có lưu history)
        self.chat_chain = RunnableWithMessageHistory(
            runnable=self.chat_prompt | self.llm | StrOutputParser(),
            get_session_history=self._get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    # ==========================================
    # QUẢN LÝ DATABASE & TRÍ NHỚ
    # ==========================================

    def _get_session_history(self, session_id: str):
        """Kết nối Postgres và cắt tỉa tin nhắn bằng gói langchain-postgres mới"""
        history = PostgresChatMessageHistory(
            "chat_history",
            session_id,
            sync_connection=self.sync_connection
        )

        # Cửa sổ trượt: Chỉ lấy 10 câu gần nhất
        if len(history.messages) > 10:
            history.messages = history.messages[-10:]

        return history

    def _save_summary(self, session_id: str, summary: str):
        """Lưu bản tóm tắt vào Postgres (UPSERT)"""
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO session_metadata (session_id, summary) 
                    VALUES (%s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET 
                        summary = EXCLUDED.summary,
                        updated_at = CURRENT_TIMESTAMP;
                """, (session_id, summary))

    def _get_summary(self, session_id: str) -> str:
        """Lấy bản tóm tắt hiện tại từ Postgres"""
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT summary FROM session_metadata WHERE session_id = %s", (session_id,))
                res = cur.fetchone()
                return res[0] if res else "Người dùng vừa mới bắt đầu phiên hội thoại."

    # ==========================================
    # CÁC HÀM XỬ LÝ CHÍNH (PUBLIC API)
    # ==========================================

    def generate_suggestion(self, session_id: str, ingredients: list[str], recipes: list[dict]) -> str:
        """Gợi ý món ăn lần đầu khi user gửi ảnh"""
        if not self.api_key: return "LLM not configured."
        
        ingredient_str = ", ".join(ingredients)
        
        # 1. LƯU GỐC: Ghi ngay danh sách nguyên liệu vào Summary để AI không bao giờ quên
        initial_summary = f"Người dùng hiện đang có các nguyên liệu: {ingredient_str}."
        self._save_summary(session_id, initial_summary)

        # 2. Chuẩn bị Context từ Database
        recipe_context = ""
        for i, r in enumerate(recipes):
            recipe_context += f"{i+1}. Tên món: {r.get('ten_mon', '')}\n"
            recipe_context += f"   - Nguyên liệu: {', '.join(r.get('nguyen_lieu_chi_tiet', []))}\n"
            recipe_context += f"   - Cách làm: {r.get('cach_lam', ['Không có'])}\n\n"

        # 3. Gọi AI
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
        """Hội thoại Follow-up có sử dụng Trí nhớ dài hạn"""
        if not self.api_key: return "LLM not configured."
        
        # 1. Kéo bản tóm tắt từ Database lên
        current_summary = self._get_summary(session_id)

        try:
            # 2. Chat với AI (Truyền cả summary và câu hỏi mới)
            response = self.chat_chain.invoke(
                {"question": message, "summary": current_summary},
                config={"configurable": {"session_id": session_id}}
            )

            # 3. CẬP NHẬT TRÍ NHỚ: Nếu chat quá 10 câu, tự động nén lại
            history_obj = self._get_session_history(session_id)
            if len(history_obj.messages) >= 10:
                # Bắt AI đọc đống lịch sử dài ngoằng và viết lại tóm tắt
                new_summary = self.summary_chain.invoke({"history": history_obj.messages})
                # Cập nhật vào DB
                self._save_summary(session_id, new_summary)
                
            return response
        except Exception as e:
            return f"Error replying to chat: {str(e)}"