import os, tempfile, shutil, threading
from fastapi import FastAPI, UploadFile, File, Form
from faster_whisper import WhisperModel

MODEL_ID = os.environ.get("MODEL_ID", "Systran/faster-whisper-small")
COMPUTE = os.environ.get("COMPUTE_TYPE", "int8")
app = FastAPI()
_model = None
_lock = threading.Lock()

def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = WhisperModel(MODEL_ID, device="cpu", compute_type=COMPUTE)
    return _model

@app.on_event("startup")
def _startup():
    threading.Thread(target=_get_model, daemon=True).start()  # 端口立即就绪，模型后台加载

@app.get("/healthz")
def healthz():
    return {"ok": True, "ready": _model is not None}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("en")):
    model = _get_model()   # 首次请求若后台还没加载完，这里同步等待
    suffix = os.path.splitext(file.filename or "audio.mp3")[1] or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        segments, info = model.transcribe(tmp.name, language=language,
                                          word_timestamps=True, vad_filter=True)
        sents, words = [], []
        for seg in segments:
            sw = []
            for w in seg.words or []:
                sw.append({"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)})
                words.append({"w": w.word, "start": round(w.start, 3), "end": round(w.end, 3)})
            sents.append({"text": seg.text.strip(), "start": round(seg.start, 3),
                          "end": round(seg.end, 3), "words": sw})
        return {"sentences": sents, "words": words,
                "duration": round(info.duration, 3), "language": info.language}
    finally:
        try: os.remove(tmp.name)
        except OSError: pass
