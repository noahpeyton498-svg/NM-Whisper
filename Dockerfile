FROM python:3.11-slim

WORKDIR /app

# ffmpeg 是 faster-whisper 解码音频的必需依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 国内网络拉 HuggingFace 模型慢时走镜像站
ENV HF_ENDPOINT=https://hf-mirror.com

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# 构建时预下载模型，避免首次请求才下载导致超时
# 想提速可改 base（快 2~3 倍）；1核2G 内存足够 small/base，medium 需 2G+ 且更慢
ARG WHISPER_MODEL=small
ENV WHISPER_MODEL=${WHISPER_MODEL}
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

EXPOSE 8000

# 单 worker：模型只加载一份；多任务靠应用内线程排队串行推理
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
