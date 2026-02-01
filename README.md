# SmartChef AI Service

SmartChef là hệ thống AI đa phương thức (multimodal), được thiết kế để phân tích nguyên liệu thông qua công nghệ Thị giác máy tính (Computer Vision) và cung cấp tư vấn ẩm thực sử dụng kiến trúc RAG (Retrieval-Augmented Generation).

## Kiến Trúc Hệ Thống (System Architecture)

Dự án tuân theo mô hình **Kiến trúc Phân tầng (Layered Architecture)** để đảm bảo tính module hóa và khả năng mở rộng:

1.  **Transport Layer**: Được triển khai bằng **FastAPI** để xử lý các HTTP request, xác thực dữ liệu đầu vào (input validation) và quản lý định tuyến (routing).
2.  **Service Layer**: Class `SmartChefService` đóng vai trò điều phối (orchestrator), quản lý luồng dữ liệu giữa các thành phần Vision, RAG và LLM.
3.  **AI Services Layer**:
    -   **Computer Vision**: Sử dụng **YOLOv8** (ONNX Runtime) cho bài toán phát hiện đối tượng (Object Detection). Được tối ưu hóa với cơ chế **In-Memory Processing** để xử lý trực tiếp luồng byte, loại bỏ độ trễ I/O ổ cứng.
    -   **RAG Engine**: Triển khai tìm kiếm ngữ nghĩa (semantic search) sử dụng **Qdrant Vector Database** và mô hình embedding **intfloat/multilingual-e5-base** (768 chiều). Tích hợp bước hậu xử lý custom sử dụng **Jaccard Similarity** để cải thiện độ phù hợp của kết quả dựa trên sự giao thoa nguyên liệu.
    -   **LLM Integration**: Tích hợp **Google Gemini 1.5 Flash** thông qua LangChain để sinh câu trả lời và quản lý ngữ cảnh phiên làm việc (session context-aware).
4.  **Infrastructure**: Các dịch vụ Vector Database (Qdrant) và Relational Database (PostgreSQL) được đóng gói và triển khai qua Docker.

## Technology Stack

-   **Runtime**: Python 3.10+
-   **Framework**: FastAPI
-   **Computer Vision**: YOLOv8, ONNX Runtime, OpenCV
-   **NLP/AI**: LangChain, Google Generative AI (Gemini), Sentence Transformers
-   **Databases**: Qdrant (Vector), PostgreSQL (Relational)
-   **Deployment**: Docker, Docker Compose

## Cài Đặt và Triển Khai

### Yêu cầu tiên quyết
-   Python 3.10 hoặc mới hơn
-   Docker Desktop

### Các bước cài đặt

1.  **Clone repository**
    ```bash
    git clone https://github.com/your-username/SmartChef-AI.git
    cd SmartChef-AI
    ```

2.  **Cấu hình Biến môi trường**
    Tạo file `.env` tại thư mục gốc:
    ```ini
    GOOGLE_API_KEY=your_gemini_key_here
    POSTGRES_USER=admin
    POSTGRES_PASSWORD=admin
    POSTGRES_DB=smartchef_db
    MODEL_NAME = gemini-2.5-flash-lite
    ```

3.  **Cài đặt các thư viện phụ thuộc**
    ```bash
    pip install -r ai_services/app/requirements.txt
    ```

4.  **Khởi động hạ tầng (Infrastructure)**
    ```bash
    docker-compose up -d
    ```

5.  **Khởi tạo Knowledge Base (RAG)**
    Chạy script ingestion để embedding và đánh index dữ liệu công thức vào Qdrant:
    ```bash
    python -m ai_services.app.RAG.prepareDataForRag.scripts.fromJsonToVectordb
    ```

6.  **Khởi chạy Application Server**
    ```bash
    python -m uvicorn ai_services.app.main:app --host 0.0.0.0 --port 8000
    ```

## Tài liệu API

Tài liệu API chi tiết có sẵn qua giao diện Swagger UI tại `http://localhost:8000/docs` khi server đang chạy.

### Các Endpoints Chính

-   **POST /api/v1/predict**
    -   **Mô tả**: Phân tích ảnh tải lên, nhận diện nguyên liệu và truy xuất gợi ý món ăn.
    -   **Input**: `multipart/form-data` (file ảnh).
    -   **Output**: JSON chứa danh sách đối tượng nhận diện, gợi ý món ăn và session ID.

-   **POST /api/v1/chat**
    -   **Mô tả**: Xử lý hội thoại tiếp nối dựa trên ngữ cảnh session đã thiết lập.
    -   **Input**: JSON body chứa `session_id` và `message`.
    -   **Output**: JSON chứa câu trả lời từ AI.
