# whisper-asr 服务（faster-whisper + FastAPI）
#
# 为什么需要异步任务接口：云托管（CloudBase Run）对单个 HTTP 请求有 60s 硬限制
# （官方《使用限制》：云托管请求超时时间 60s，不可配置），长音频同步转写必被
# 网关掐断（503/504）。因此提供「提交任务秒回 taskId -> 轮询查询结果」的异步模式，
# 每个 HTTP 请求都远小于 60s，转写时长不受限制。
#
# 接口：
#   GET  /healthz       健康检查 -> {"ok": true}
#   POST /task_submit   multipart 上传音频（字段 file，可选 language，默认 en）
#                       -> 秒回 {"taskId": "..."}
#   GET  /task?id=...   查询任务 -> {"status": "queued|running|done|error",
#                                   "result": <transcribe 返回体>, "error": "..."}
#   POST /transcribe    同步转写（保留兼容，短音频可用；长音频会撞 60s 网关限制）
#
# 返回体（与旧版完全一致，前端/云函数无需改动解析）：
#   {"sentences":[{"text","start","end","words":[{"word","start","end"}]}],
#    "words":[{"w","start","end"}], "duration", "language"}

import os
import time
import uuid
import threading
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from faster_whisper import WhisperModel

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
TASK_TTL_SEC = 30 * 60          # 完成的任务结果保留 30 分钟，之后清理

app = FastAPI()

_model = None
_model_lock = threading.Lock()      # 模型初始化锁
_infer_lock = threading.Lock()      # 推理锁：faster-whisper 不支持同模型并发推理，串行排队
_tasks = {}                         # task_id -> {"status","result","error","created","finished"}
_tasks_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def transcribe_file(path, language):
    """转写一个音频文件。faster-whisper 的 transcribe 返回生成器，实际推理在迭代时发生，
    因此迭代必须在推理锁内完成（多任务排队串行执行）。"""
    model = get_model()
    sentences = []
    words = []
    with _infer_lock:
        segments, info = model.transcribe(
            path,
            language=(language or None),
            word_timestamps=True,   # 关键：词级时间戳
            vad_filter=True,        # VAD 过滤静音，提升词级时间戳质量
        )
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            sw = []
            for w in (seg.words or []):
                t = (w.word or "").strip()
                if t and w.end > w.start:
                    sw.append({"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)})
                    words.append({"w": w.word, "start": round(w.start, 3), "end": round(w.end, 3)})
            sentences.append({
                "text": text,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "words": sw,
            })
    if not sentences:
        raise ValueError("未识别出任何语音内容")
    return {
        "sentences": sentences,
        "words": words,
        "duration": round(float(info.duration or 0), 3),
        "language": info.language,
    }


def run_task(task_id, path, language):
    with _tasks_lock:
        _tasks[task_id]["status"] = "running"
    try:
        result = transcribe_file(path, language)
        with _tasks_lock:
            _tasks[task_id].update(status="done", result=result, finished=time.time())
    except Exception as err:   # 转写失败不影响服务本身，状态标记 error 供前端提示
        with _tasks_lock:
            _tasks[task_id].update(status="error", error=str(err)[:500], finished=time.time())
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    gc_tasks()


def gc_tasks():
    now = time.time()
    with _tasks_lock:
        for tid in list(_tasks.keys()):
            fin = _tasks[tid].get("finished")
            if fin and now - fin > TASK_TTL_SEC:
                _tasks.pop(tid, None)


@app.get("/healthz")
def healthz():
    return {"ok": True}


async def _save_upload(file: UploadFile):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    suffix = os.path.splitext(file.filename or "")[1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return tmp.name


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("en")):
    path = await _save_upload(file)
    try:
        return transcribe_file(path, language)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/task_submit")
async def task_submit(file: UploadFile = File(...), language: str = Form("en")):
    path = await _save_upload(file)
    task_id = uuid.uuid4().hex
    with _tasks_lock:
        _tasks[task_id] = {"status": "queued", "created": time.time()}
    threading.Thread(target=run_task, args=(task_id, path, language), daemon=True).start()
    return {"taskId": task_id}


@app.get("/task")
def task_get(id: str):
    with _tasks_lock:
        t = dict(_tasks.get(id) or {})
    if not t:
        raise HTTPException(status_code=404, detail="task not found")
    out = {"status": t.get("status")}
    if t.get("status") == "done":
        out["result"] = t.get("result")
    if t.get("error"):
        out["error"] = t.get("error")
    return out
