FROM python:3.11-slim

# 安装 OpenCV 和 Faiss 所需的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先安装依赖（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建运行时目录
RUN mkdir -p uploads static/output

# Hugging Face Spaces 要求监听 7860 端口
EXPOSE 7860

CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "4", "--timeout", "300", "-b", "0.0.0.0:7860"]
