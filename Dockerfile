FROM python:3.11-slim

# 安装 OpenCV(contrib/GUI版)、mediapipe、ffmpeg 所需的系统依赖
# libgl1/libsm6/libxext6/libxrender1: opencv-contrib-python(GUI版) 必需
# libgomp1: scipy 数值库; libglib2.0-0: gstreamer/glib 基础
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先安装依赖（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建运行时目录
RUN mkdir -p uploads static/output

# 端口由环境变量 PORT 控制（默认 5000；HuggingFace 可传 7860）
EXPOSE 5000

CMD ["sh", "-c", "gunicorn app:app --workers 2 --threads 4 --timeout 300 -b 0.0.0.0:${PORT:-5000}"]
