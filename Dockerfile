FROM python:3.11-slim
ENV HF_ENDPOINT=https://hf-mirror.com \
    MODEL_ID=Systran/faster-whisper-small \
    COMPUTE_TYPE=int8 \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart faster-whisper huggingface_hub
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['MODEL_ID'])"
WORKDIR /app
COPY server.py /app/server.py
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
