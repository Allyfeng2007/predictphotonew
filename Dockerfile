# Dockerfile
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV 运行所需基础库 + CA 证书
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装 CPU 版 PyTorch 和 torchvision（关键！不能走国内镜像）
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1

# 再装其余依赖（如需可换清华镜像）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


# 拷贝代码与模型
COPY app.py .
COPY yolov8n.pt .

EXPOSE 80

# 云托管要求监听 80 端口
CMD ["gunicorn", "-b", "0.0.0.0:80", "-w", "2", "-k", "gthread", "--threads", "4", "--timeout", "120", "app:app"]
