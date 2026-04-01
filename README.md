# SmartChef AI Service

SmartChef là hệ thống AI đa phương thức (multimodal), được thiết kế để phân tích nguyên liệu thông qua công nghệ Thị giác máy tính (Computer Vision) và cung cấp tư vấn ẩm thực sử dụng kiến trúc RAG (Retrieval-Augmented Generation).

## Kiến Trúc Hệ Thống (System Architecture)

Dự án tuân theo mô hình **Kiến trúc Phân tầng (Layered Architecture)** để đảm bảo tính module hóa và khả năng mở rộng:

1.  **Transport Layer**: Được triển khai bằng **FastAPI** để xử lý các HTTP request, xác thực dữ liệu đầu vào (input validation) và quản lý định tuyến (routing).
2.  **Service Layer**: Class `SmartChefService` đóng vai trò điều phối (orchestrator), quản lý luồng dữ liệu giữa các thành phần Vision, RAG và LLM.
3.  **AI Services Layer**:
    -   **Computer Vision**: Sử dụng **YOLOv8** (ONNX Runtime) cho bài toán phát hiện đối tượng (Object Detection). Được tối ưu hóa với cơ chế **In-Memory Processing** để xử lý trực tiếp luồng byte, loại bỏ độ trễ I/O ổ cứng.
    -   **RAG Engine**: Triển khai tìm kiếm ngữ nghĩa (semantic search) sử dụng **Qdrant Vector Database** và mô hình embedding **intfloat/multilingual-e5-base** (768 chiều). Tích hợp bước hậu xử lý custom sử dụng **Jaccard Similarity** để cải thiện độ phù hợp của kết quả dựa trên sự giao thoa nguyên liệu.
    -   **LLM Integration**: Tích hợp **Google Gemini 2.0 Flash** thông qua LangChain để sinh câu trả lời và quản lý ngữ cảnh phiên làm việc (session context-aware).
4.  **Infrastructure**: Các dịch vụ Vector Database (Qdrant) và Relational Database (PostgreSQL) được đóng gói và triển khai qua Docker.

## Technology Stack

-   **Runtime**: Python 3.10+
-   **Framework**: FastAPI
-   **Computer Vision**: YOLOv8, ONNX Runtime, OpenCV
-   **NLP/AI**: LangChain, Google Generative AI (Gemini), Sentence Transformers
-   **Databases**: Qdrant (Vector), PostgreSQL (Relational)
-   **Deployment**: Docker, Docker Compose

## Cài Đặt và Triển Khai (3 Bước Siêu Tốc)

Dự án đã được "Docker hóa" toàn phần. Bạn không cần cài đặt Python hay database thủ công.

### 1. Clone repository
```bash
git clone https://github.com/your-username/SmartChef-AI.git
cd SmartChef-AI
```

### 2. Cấu hình Biến môi trường
Sao chép file mẫu và điền `GOOGLE_API_KEY` của bạn:
```bash
cp .env.example .env
# Mở file .env và dán API Key của bạn vào
```

### 3. Khởi chạy hệ thống
Chỉ cần một lệnh duy nhất để khởi động toàn bộ Database, Vector DB và AI Service:
```bash
docker-compose up --build -d
```

> **Lưu ý**: Hệ thống sẽ tự động kiểm tra và nạp dữ liệu (Ingestion) vào database trong lần đầu khởi chạy. Bạn có thể theo dõi quá trình này qua log: `docker-compose logs -f ai_app`.

## Tài liệu API

Tài liệu API chi tiết (Swagger UI) sẽ sẵn sàng tại `http://localhost:8000/docs` ngay sau khi hệ thống khởi động xong.

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
