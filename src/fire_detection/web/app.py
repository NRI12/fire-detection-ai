import asyncio
import base64
import json
import tempfile
import time
import urllib.request
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from fire_detection.inference.model import YOLOv8Detector
from fire_detection.web.email_alert import GmailSender
from fire_detection.web.gradcam import YOLOEigenCAM

BASE_DIR = Path(__file__).resolve().parents[3]
WEIGHTS = BASE_DIR / "runs/detect/runs/train/dfire_v1_s_baseline/weights/best.pt"
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = STATIC_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = Path(__file__).parent / "templates"
SETTINGS_HTML = TEMPLATES_DIR / "settings.html"

detector: YOLOv8Detector | None = None
gradcam: YOLOEigenCAM | None = None
history: deque = deque(maxlen=50)
gmail_sender = GmailSender()
video_tasks: dict[str, Path] = {}

CLASS_COLORS = {
    "smoke": (128, 128, 128),
    "fire":  (0, 60, 255),
}

app = FastAPI(title="Fire Detection AI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
INDEX_HTML = TEMPLATES_DIR / "index.html"


@app.on_event("startup")
async def _load_model():
    global detector, gradcam
    detector = YOLOv8Detector(str(WEIGHTS))
    gradcam = YOLOEigenCAM(detector._model, device=detector.device)


def _img_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()


def _draw_colored(frame: np.ndarray, detections) -> np.ndarray:
    annotated = frame.copy()
    h, w = frame.shape[:2]
    thickness = max(2, int(min(h, w) / 300))
    font_scale = max(0.5, min(h, w) / 900)

    for det in detections:
        x1, y1, x2, y2 = det.box
        color = CLASS_COLORS.get(det.class_name, (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(annotated, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
        cv2.putText(annotated, label, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def _run_pipeline(frame_bgr: np.ndarray, conf: float = 0.45) -> dict:
    detections, elapsed = detector.predict(frame_bgr, conf=conf)
    annotated = _draw_colored(frame_bgr, detections)
    cam_img = gradcam.generate(frame_bgr)

    return {
        "detections": [
            {
                "class_name": d.class_name,
                "confidence": round(d.confidence, 3),
                "box": list(d.box),
            }
            for d in detections
        ],
        "inference_ms": round(elapsed * 1000, 1),
        "input_b64": _img_to_b64(frame_bgr),
        "output_b64": _img_to_b64(annotated),
        "gradcam_b64": _img_to_b64(cam_img),
    }


def _decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Không thể đọc ảnh. Định dạng không được hỗ trợ.")
    return img


def _maybe_send_alert(background_tasks: BackgroundTasks | None, result: dict, source: str):
    if not result["detections"] or not gmail_sender.is_configured():
        return
    if background_tasks is not None:
        background_tasks.add_task(
            gmail_sender.send_alert,
            result["detections"],
            result["output_b64"],
            result["gradcam_b64"],
            source,
            result["inference_ms"],
        )


def _save_history(result: dict, source: str):
    history.appendleft({
        "timestamp": time.strftime("%H:%M:%S"),
        "source": source,
        "count": len(result["detections"]),
        "inference_ms": result["inference_ms"],
        "detections": result["detections"],
        "thumbnail_b64": result["output_b64"],
    })


@app.get("/")
async def index():
    return FileResponse(str(INDEX_HTML), media_type="text/html",
                        headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


def _process_video(video_path: Path, conf: float, sample_rate: int) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Không thể mở video. Định dạng không được hỗ trợ.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w == 0 or h == 0:
        cap.release()
        raise ValueError("Không đọc được kích thước video.")

    out_name = f"result_{uuid.uuid4().hex[:8]}.mp4"
    out_path = RESULTS_DIR / out_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    last_detections: list = []
    detections_summary: list = []
    inference_times: list = []
    thumbnail_b64: str | None = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            dets, elapsed = detector.predict(frame, conf=conf)
            last_detections = dets
            inference_times.append(elapsed * 1000)
            if dets:
                detections_summary.append({
                    "frame": frame_idx,
                    "timestamp_s": round(frame_idx / fps, 2),
                    "detections": [
                        {"class_name": d.class_name, "confidence": round(d.confidence, 3)}
                        for d in dets
                    ],
                })
                if thumbnail_b64 is None:
                    thumbnail_b64 = _img_to_b64(_draw_colored(frame, dets))

        annotated = _draw_colored(frame, last_detections)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    if frame_idx == 0:
        out_path.unlink(missing_ok=True)
        raise ValueError("Video rỗng hoặc không đọc được frame nào.")

    fire_frames = sum(
        1 for s in detections_summary
        if any(d["class_name"] == "fire" for d in s["detections"])
    )
    smoke_frames = sum(
        1 for s in detections_summary
        if any(d["class_name"] == "smoke" for d in s["detections"])
    )

    return {
        "video_url": f"/static/results/{out_name}",
        "total_frames": frame_idx,
        "fps": round(fps, 2),
        "duration_s": round(frame_idx / fps, 2),
        "processed_frames": len(inference_times),
        "inference_ms_avg": round(sum(inference_times) / len(inference_times), 1) if inference_times else 0,
        "detections_summary": detections_summary,
        "fire_frames": fire_frames,
        "smoke_frames": smoke_frames,
        "thumbnail_b64": thumbnail_b64,
    }


@app.post("/api/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    conf: float = Form(0.45),
    sample_rate: int = Form(1),
    background_tasks: BackgroundTasks = None,
):
    data = await file.read()
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        result = await asyncio.to_thread(_process_video, tmp_path, conf, sample_rate)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    history.appendleft({
        "timestamp": time.strftime("%H:%M:%S"),
        "source": "video",
        "count": sum(len(s["detections"]) for s in result["detections_summary"]),
        "inference_ms": result["inference_ms_avg"],
        "detections": [],
        "thumbnail_b64": result["thumbnail_b64"] or "",
    })

    if result["fire_frames"] > 0 and background_tasks is not None and gmail_sender.is_configured():
        first_det = result["detections_summary"][0]["detections"] if result["detections_summary"] else []
        background_tasks.add_task(
            gmail_sender.send_alert,
            first_det,
            result.get("thumbnail_b64", ""),
            "",
            "video",
            result["inference_ms_avg"],
        )

    return JSONResponse(result)


@app.post("/api/predict/upload")
async def predict_upload(
    file: UploadFile = File(...),
    conf: float = Form(0.45),
    background_tasks: BackgroundTasks = None,
):
    data = await file.read()
    frame = _decode_upload(data)
    result = _run_pipeline(frame, conf)
    _save_history(result, source="upload")
    _maybe_send_alert(background_tasks, result, source="upload")
    return JSONResponse(result)


@app.post("/api/predict/url")
async def predict_url(
    url: str = Form(...),
    conf: float = Form(0.45),
    background_tasks: BackgroundTasks = None,
):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
    except Exception as e:
        raise HTTPException(400, f"Không thể tải ảnh từ URL: {e}")
    frame = _decode_upload(data)
    result = _run_pipeline(frame, conf)
    _save_history(result, source="url")
    _maybe_send_alert(background_tasks, result, source="url")
    return JSONResponse(result)


@app.post("/api/predict/sample")
async def predict_sample(
    filename: str = Form(...),
    conf: float = Form(0.45),
    background_tasks: BackgroundTasks = None,
):
    path = STATIC_DIR / "samples" / filename
    if not path.exists():
        raise HTTPException(404, "Không tìm thấy ảnh mẫu.")
    frame = cv2.imread(str(path))
    if frame is None:
        raise HTTPException(500, "Không thể đọc ảnh mẫu.")
    result = _run_pipeline(frame, conf)
    _save_history(result, source=f"sample:{filename}")
    _maybe_send_alert(background_tasks, result, source=f"sample:{filename}")
    return JSONResponse(result)


@app.get("/api/samples")
async def list_samples():
    samples_dir = STATIC_DIR / "samples"
    files = sorted(samples_dir.glob("*.jpg")) + sorted(samples_dir.glob("*.png"))
    return {"samples": [f.name for f in files]}


@app.get("/api/history")
async def get_history():
    return {"history": list(history)}


@app.get("/api/model-info")
async def model_info():
    return {
        "weights": WEIGHTS.name if WEIGHTS.exists() else "N/A",
        "classes": ["smoke", "fire"],
        "device": detector.device if detector else "unknown",
        "framework": "YOLOv8",
    }


@app.get("/settings")
async def settings_page():
    return FileResponse(str(SETTINGS_HTML), media_type="text/html")


@app.get("/api/settings")
async def get_settings():
    cfg = gmail_sender.load_config()
    return {
        "sender_email": cfg.get("sender_email", ""),
        "app_password": "***" if cfg.get("app_password") else "",
        "recipient_email": cfg.get("recipient_email", ""),
        "configured": gmail_sender.is_configured(),
    }


@app.post("/api/settings")
async def save_settings(request: Request):
    body = await request.json()
    allowed = {"sender_email", "app_password", "recipient_email"}
    updates = {k: v for k, v in body.items() if k in allowed and v != "***"}
    gmail_sender.save_config(updates)
    return {"ok": True}


@app.post("/api/settings/test")
async def test_connection():
    ok, message = gmail_sender.test_connection()
    return {"ok": ok, "message": message}


@app.post("/api/stream/video/upload")
async def upload_video_for_stream(file: UploadFile = File(...)):
    data = await file.read()
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    cap = cv2.VideoCapture(str(tmp_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    task_id = uuid.uuid4().hex[:12]
    video_tasks[task_id] = tmp_path
    return {"task_id": task_id, "total_frames": total_frames, "fps": round(fps, 1)}


@app.get("/api/stream/video/{task_id}")
async def stream_video(task_id: str, conf: float = 0.45):
    tmp_path = video_tasks.pop(task_id, None)
    if not tmp_path or not tmp_path.exists():
        raise HTTPException(404, "Task not found")

    INFERENCE_EVERY = 3   # run model every 3 displayed frames
    TARGET_DISPLAY_FPS = 15  # cap display at 15 fps to reduce bandwidth

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=80)

    def worker():
        try:
            cap = cv2.VideoCapture(str(tmp_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            # Skip source frames to hit target display fps
            src_skip = max(1, round(fps / TARGET_DISPLAY_FPS))
            frame_idx = 0       # source frame counter
            display_idx = 0     # frames actually sent to client
            last_dets: list = []
            fire_frames = 0
            smoke_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % src_skip == 0:
                    # Run inference every INFERENCE_EVERY displayed frames
                    if display_idx % INFERENCE_EVERY == 0:
                        dets, _ = detector.predict(frame, conf=conf)
                        last_dets = dets
                        if any(d.class_name == "fire" for d in last_dets):
                            fire_frames += 1
                        if any(d.class_name == "smoke" for d in last_dets):
                            smoke_frames += 1

                    annotated = _draw_colored(frame, last_dets)
                    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 62])
                    b64 = base64.b64encode(buf).decode()
                    new_inf = (display_idx % INFERENCE_EVERY == 0)
                    payload = json.dumps({
                        "t": "f",
                        "i": frame_idx,
                        "ts": round(frame_idx / fps, 1),
                        "b": b64,
                        "d": [{"c": d.class_name, "p": round(d.confidence, 2)} for d in last_dets] if new_inf else [],
                        "fire": fire_frames,
                        "smoke": smoke_frames,
                    })
                    try:
                        asyncio.run_coroutine_threadsafe(queue.put(payload), loop).result(timeout=15)
                    except Exception:
                        break
                    display_idx += 1

                frame_idx += 1

            cap.release()
            done = json.dumps({
                "t": "done",
                "total": frame_idx,
                "fps": round(fps, 1),
                "dur": round(frame_idx / fps, 1),
                "fire": fire_frames,
                "smoke": smoke_frames,
            })
            try:
                asyncio.run_coroutine_threadsafe(queue.put(done), loop).result(timeout=5)
            except Exception:
                pass
        except Exception as exc:
            try:
                asyncio.run_coroutine_threadsafe(
                    queue.put(json.dumps({"t": "error", "msg": str(exc)})), loop
                ).result(timeout=5)
            except Exception:
                pass
        finally:
            tmp_path.unlink(missing_ok=True)

    loop.run_in_executor(None, worker)

    async def generate():
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=120)
            except asyncio.TimeoutError:
                break
            yield f"data: {item}\n\n"
            if json.loads(item).get("t") in ("done", "error"):
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
