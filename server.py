"""
FastAPI inference server for handwritten text recognition.

Endpoints:
  POST /ocr        — one image, returns transcript + per-word details.
  POST /ocr/batch  — several images in one request.
  GET  /health     — liveness + engine/pool state.

Design notes:
  * PaddleOCR predictors are NOT thread-safe, so the server keeps a small pool
    of engines; a request borrows one for the duration of inference and
    returns it. Pool size, not thread count, is the real concurrency limit.
  * Decoding, resizing and inference all happen on the pool's threads, so the
    async event loop is never blocked — not even by a 12 MP JPEG decode.
  * Requests beyond the pool's capacity wait briefly, then get 503 with
    Retry-After instead of piling up unbounded.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Iterator

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ocr_server")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


_CPU = os.cpu_count() or 4

USE_GPU: bool = _env_bool("USE_GPU", "true")
DET_MODEL_DIR: str | None = os.environ.get("DET_MODEL_DIR") or None
REC_MODEL_DIR: str | None = os.environ.get("REC_MODEL_DIR") or None
REC_CHAR_DICT: str = os.environ.get("REC_CHAR_DICT", "./dict/en_dict.txt")
CONF_THRESHOLD: float = float(os.environ.get("CONF_THRESHOLD", "0.5"))
# One engine per concurrent inference. GPU memory is the binding constraint on
# GPU; RAM and core count on CPU. More than this buys queueing, not throughput.
POOL_SIZE: int = int(os.environ.get("POOL_SIZE", "1" if USE_GPU else str(min(2, _CPU))))
CPU_THREADS: int = int(os.environ.get("CPU_THREADS", str(max(1, min(_CPU // max(POOL_SIZE, 1), 8)))))
QUEUE_TIMEOUT: float = float(os.environ.get("QUEUE_TIMEOUT", "20"))
MAX_QUEUED: int = int(os.environ.get("MAX_QUEUED", str(POOL_SIZE * 8)))
MAX_UPLOAD_MB: float = float(os.environ.get("MAX_UPLOAD_MB", "20"))
MAX_SIDE: int = int(os.environ.get("MAX_SIDE", "1600"))  # 0 disables downscaling
MAX_BATCH: int = int(os.environ.get("MAX_BATCH", "16"))
CORS_ORIGINS: list[str] = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
]

_MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

# OpenCV runs its own pool inside every worker thread; keep it modest so the
# decode threads and the inference runtime don't oversubscribe the CPU.
cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, min(2, _CPU)))

# ---------------------------------------------------------------------------
# Engine pool
# ---------------------------------------------------------------------------
def build_engine():
    """Construct one PaddleOCR predictor. Called once per pool slot."""
    if not USE_GPU:
        os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
        os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))

    from paddleocr import PaddleOCR  # imported lazily, after the env vars above

    kwargs: dict[str, Any] = {
        "use_textline_orientation": True,
        "lang": "en",
        "device": "gpu" if USE_GPU else "cpu",
        "text_det_thresh": 0.3,
        "text_det_box_thresh": 0.5,
        "text_det_unclip_ratio": 2.0,
    }
    if DET_MODEL_DIR:
        kwargs["text_detection_model_dir"] = DET_MODEL_DIR
    if REC_MODEL_DIR:
        kwargs["text_recognition_model_dir"] = REC_MODEL_DIR
    if REC_CHAR_DICT and os.path.isfile(REC_CHAR_DICT):
        kwargs["character_dict_path"] = REC_CHAR_DICT
    if not USE_GPU:
        kwargs["cpu_threads"] = CPU_THREADS
        kwargs["enable_mkldnn"] = True

    try:
        engine = PaddleOCR(**kwargs)
    except TypeError:
        for key in ("cpu_threads", "enable_mkldnn", "character_dict_path"):
            kwargs.pop(key, None)
        engine = PaddleOCR(**kwargs)

    # Warm the lazy graph so the first real request isn't the slow one.
    with contextlib.suppress(Exception):
        engine.predict(np.full((320, 320, 3), 255, np.uint8))
    return engine


class EnginePool:
    """Lazily grown, fixed-size pool of non-thread-safe predictors."""

    def __init__(self, factory, size: int) -> None:
        self._factory = factory
        self._size = max(1, size)
        self._idle: queue.LifoQueue = queue.LifoQueue()
        self._lock = threading.Lock()
        self._created = 0

    @property
    def created(self) -> int:
        return self._created

    @property
    def idle(self) -> int:
        return self._idle.qsize()

    def prime(self) -> None:
        """Build the first engine eagerly so startup fails loudly, not lazily."""
        self._idle.put(self._new())

    def _new(self):
        with self._lock:
            if self._created >= self._size:
                return None
            self._created += 1
        try:
            return self._factory()
        except Exception:
            with self._lock:
                self._created -= 1
            raise

    @contextlib.contextmanager
    def borrow(self, timeout: float) -> Iterator[Any]:
        try:
            engine = self._idle.get_nowait()
        except queue.Empty:
            engine = self._new()
            if engine is None:
                try:
                    engine = self._idle.get(timeout=timeout)
                except queue.Empty:
                    raise TimeoutError("no OCR engine available") from None
        try:
            yield engine
        finally:
            self._idle.put(engine)


_pool: EnginePool | None = None
_executor: ThreadPoolExecutor | None = None
_slots: asyncio.Semaphore | None = None
_started_at = time.time()


# ---------------------------------------------------------------------------
# Inference (runs on pool threads)
# ---------------------------------------------------------------------------
def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image data — unsupported format or corrupt file.")
    return img


def _parse(raw: Any, inv_scale: float, want_bbox: bool) -> list[dict[str, Any]]:
    """Normalise PaddleOCR 3.x dict output and the legacy nested-list output."""
    if not raw:
        return []
    result = raw[0] if isinstance(raw, (list, tuple)) else raw
    pairs: list[tuple[str, float, Any]] = []

    if isinstance(result, dict) and "rec_texts" in result:
        polys = result.get("rec_polys")
        if polys is None:
            polys = result.get("dt_polys") or ()
        pairs = list(zip(result.get("rec_texts") or (), result.get("rec_scores") or (), polys))
    else:
        for line in (result or ()):
            if line:
                bbox, (text, conf) = line
                pairs.append((text, conf, bbox))

    words: list[dict[str, Any]] = []
    for text, conf, poly in pairs:
        if conf is None:
            continue
        conf = float(conf)
        if conf < CONF_THRESHOLD:
            continue
        entry: dict[str, Any] = {"text": text, "confidence": round(conf, 4)}
        if want_bbox:
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if inv_scale != 1.0:
                pts = pts * inv_scale
            # Vectorised rounding beats a Python loop over every corner point.
            entry["bbox"] = np.round(pts, 1).tolist()
        words.append(entry)
    return words


def _process(data: bytes, want_bbox: bool) -> dict[str, Any]:
    """Decode → downscale → OCR → parse. The whole request's CPU work."""
    if _pool is None:
        raise RuntimeError("OCR engine is not available (startup failed).")

    t0 = time.perf_counter()
    img = _decode(data)
    height, width = img.shape[:2]

    scale = 1.0
    if MAX_SIDE > 0 and max(height, width) > MAX_SIDE:
        scale = MAX_SIDE / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    decode_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    with _pool.borrow(QUEUE_TIMEOUT) as engine:
        raw = engine.predict(img)
    infer_s = time.perf_counter() - t1

    words = _parse(raw, 1.0 / scale if scale else 1.0, want_bbox)
    avg = sum(w["confidence"] for w in words) / len(words) if words else 0.0
    return {
        "text": " ".join(w["text"] for w in words),
        "words": words,
        "avg_confidence": round(avg, 4),
        "image_size": [width, height],
        "decode_time_s": round(decode_s, 4),
        "inference_time_s": round(infer_s, 4),
    }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool, _executor, _slots

    logger.info("Starting OCR server...")
    for key, value in (
        ("use_gpu", USE_GPU), ("det_model_dir", DET_MODEL_DIR or "(default PP-OCRv4)"),
        ("rec_model_dir", REC_MODEL_DIR or "(default PP-OCRv4)"),
        ("rec_char_dict", REC_CHAR_DICT), ("conf_threshold", CONF_THRESHOLD),
        ("pool_size", POOL_SIZE), ("cpu_threads", CPU_THREADS),
        ("max_side", MAX_SIDE), ("max_upload_mb", MAX_UPLOAD_MB),
    ):
        logger.info("  %-15s: %s", key, value)

    _executor = ThreadPoolExecutor(max_workers=POOL_SIZE, thread_name_prefix="ocr")
    _slots = asyncio.Semaphore(MAX_QUEUED)

    pool = EnginePool(build_engine, POOL_SIZE)
    loop = asyncio.get_running_loop()
    try:
        # Build (and warm) the first engine off the loop so startup probes answer.
        await loop.run_in_executor(_executor, pool.prime)
        _pool = pool
        logger.info("PaddleOCR engine loaded and warmed up.")
    except Exception as exc:
        logger.error("Failed to load PaddleOCR: %s", exc)
        _pool = None  # /ocr returns 503 until this is fixed

    yield

    logger.info("Shutting down OCR server...")
    if _executor:
        _executor.shutdown(wait=True)
    logger.info("Server shut down cleanly.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Handwritten OCR API",
    description="PaddleOCR-based handwritten text recognition service",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
_index_html = os.path.join(_frontend_dir, "index.html")
_has_frontend = os.path.isdir(_frontend_dir)
if _has_frontend:
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    if _has_frontend and os.path.isfile(_index_html):
        return FileResponse(_index_html)
    return JSONResponse({"detail": "Frontend not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------
_ALLOWED_TYPES = frozenset(
    {"image/jpeg", "image/jpg", "image/png", "image/bmp",
     "image/tiff", "image/webp", "application/octet-stream", ""}
)


async def _read_upload(file: UploadFile) -> bytes:
    """Stream the upload with a hard size cap, so a huge POST can't OOM us."""
    ctype = (file.content_type or "").split(";")[0].strip().lower()
    if ctype and ctype not in _ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported media type: {ctype}. "
                                 "Send JPEG, PNG, BMP, TIFF, or WebP.")
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1 << 20):
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Image exceeds {MAX_UPLOAD_MB:g} MB limit.")
        chunks.append(chunk)
    if not total:
        raise HTTPException(400, "Empty upload.")
    return b"".join(chunks)


async def _run(data: bytes, want_bbox: bool) -> dict[str, Any]:
    """Admission control, then hand the work to a pool thread."""
    if _pool is None or _executor is None or _slots is None:
        raise HTTPException(503, "OCR engine not available.")
    try:
        await asyncio.wait_for(_slots.acquire(), timeout=QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(503, "Server busy, retry shortly.",
                            headers={"Retry-After": "5"}) from None
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, _process, data, want_bbox)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except TimeoutError:
        raise HTTPException(503, "Inference queue timed out.",
                            headers={"Retry-After": "5"}) from None
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from None
    except Exception as exc:
        logger.exception("OCR inference error")
        raise HTTPException(500, f"Inference error: {exc}") from None
    finally:
        _slots.release()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Liveness check")
async def health():
    ready = _pool is not None
    body = {
        "status": "ok" if ready else "degraded",
        "engine_ready": ready,
        "uptime_s": round(time.time() - _started_at, 1),
        "pool": {
            "size": POOL_SIZE,
            "created": _pool.created if _pool else 0,
            "idle": _pool.idle if _pool else 0,
        },
        "device": "gpu" if USE_GPU else "cpu",
    }
    return JSONResponse(body, status_code=200 if ready else 503)


@app.post("/ocr", summary="Recognise handwritten text in an uploaded image")
async def ocr_endpoint(
    file: UploadFile = File(...),
    include_bbox: bool = Query(True, description="Set false for a much smaller response."),
):
    """
    **Request**: multipart/form-data with field ``file``.

    **Response** (200):
    ```json
    {
      "text": "full transcript",
      "words": [{"text": "word", "confidence": 0.95, "bbox": [[x, y], ...]}],
      "avg_confidence": 0.93,
      "image_size": [w, h],
      "decode_time_s": 0.01,
      "inference_time_s": 0.12
    }
    ```
    """
    data = await _read_upload(file)
    result = await _run(data, include_bbox)
    logger.info("OCR: %d words, avg_conf=%.3f, decode=%.3fs, infer=%.3fs",
                len(result["words"]), result["avg_confidence"],
                result["decode_time_s"], result["inference_time_s"])
    return result


@app.post("/ocr/batch", summary="Recognise text in several images in one request")
async def ocr_batch(
    files: list[UploadFile] = File(...),
    include_bbox: bool = Query(True),
):
    """Same as /ocr, but N images per round-trip. Failures are per-item, not fatal."""
    if not files:
        raise HTTPException(400, "No files uploaded.")
    if len(files) > MAX_BATCH:
        raise HTTPException(413, f"At most {MAX_BATCH} images per batch.")

    async def one(f: UploadFile) -> dict[str, Any]:
        name = f.filename or "upload"
        try:
            data = await _read_upload(f)
            return {"filename": name, "ok": True, **await _run(data, include_bbox)}
        except HTTPException as exc:
            return {"filename": name, "ok": False, "error": exc.detail,
                    "status": exc.status_code}

    started = time.perf_counter()
    results = await asyncio.gather(*(one(f) for f in files))
    return {
        "count": len(results),
        "succeeded": sum(1 for r in results if r["ok"]),
        "total_time_s": round(time.perf_counter() - started, 4),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        workers=1,  # concurrency comes from POOL_SIZE, not from process forks
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        reload=False,
    )
