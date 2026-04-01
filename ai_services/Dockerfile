# Sử dụng Python 3.10 slim để giảm dung lượng image
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và ONNX
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Thiết lập biến môi trường Python
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Port mặc định của FastAPI
EXPOSE 8000

# Lệnh khởi chạy server (Sử dụng uvicorn)
# Chú ý: main.py nằm trong thư mục app/
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
