"""
FastAPI inference server for handwritten text recognition.

Exposes:
  POST /ocr  — accepts an uploaded image, returns JSON with recognised text,
               per-word details, and confidence scores.
  GET  /health — liveness check.

The PaddleOCR engine is loaded once at startup inside a lifespan context
manager.  Heavy inference is dispatched to a ThreadPoolExecutor so that the
async event loop is never blocked.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
"""

import io
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ocr_server")

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

USE_GPU: bool = os.environ.get("USE_GPU", "true").lower() in ("1", "true", "yes")
DET_MODEL_DIR: Optional[str] = os.environ.get("DET_MODEL_DIR", None)
REC_MODEL_DIR: Optional[str] = os.environ.get("REC_MODEL_DIR", None)
REC_CHAR_DICT: str = os.environ.get("REC_CHAR_DICT", "./dict/en_dict.txt")
CONF_THRESHOLD: float = float(os.environ.get("CONF_THRESHOLD", "0.5"))
MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", "4"))

# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------

_ocr_engine = None
_executor: Optional[ThreadPoolExecutor] = None


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load OCR engine and create thread pool at startup; clean up at shutdown."""
    global _ocr_engine, _executor

    logger.info("Starting OCR server...")
    logger.info(f"  use_gpu        : {USE_GPU}")
    logger.info(f"  det_model_dir  : {DET_MODEL_DIR or '(default PP-OCRv4)'}")
    logger.info(f"  rec_model_dir  : {REC_MODEL_DIR or '(default PP-OCRv4)'}")
    logger.info(f"  rec_char_dict  : {REC_CHAR_DICT}")
    logger.info(f"  conf_threshold : {CONF_THRESHOLD}")
    logger.info(f"  max_workers    : {MAX_WORKERS}")

    try:
        from paddleocr import PaddleOCR

        kwargs: Dict[str, Any] = dict(
            use_textline_orientation=True,
            lang="en",
            device="gpu" if USE_GPU else "cpu",
            text_det_thresh=0.3,
            text_det_box_thresh=0.5,
            text_det_unclip_ratio=2.0,
        )
        if DET_MODEL_DIR:
            kwargs["text_detection_model_dir"] = DET_MODEL_DIR
        if REC_MODEL_DIR:
            kwargs["text_recognition_model_dir"] = REC_MODEL_DIR

        _ocr_engine = PaddleOCR(**kwargs)
        logger.info("PaddleOCR engine loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load PaddleOCR: {exc}")
        # Server will still start — /ocr endpoint will return 503 until fixed.
        _ocr_engine = None

    _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    logger.info("Thread pool executor created.")

    yield  # ---- application runs here ----

    logger.info("Shutting down OCR server...")
    if _executor:
        _executor.shutdown(wait=True)
    logger.info("Server shut down cleanly.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Handwritten OCR API",
    description="PaddleOCR-based handwritten text recognition service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
_frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = os.path.join(_frontend_dir, "index.html")
    return FileResponse(index) if os.path.isfile(index) else JSONResponse({"detail": "Frontend not found"}, 404)


# ---------------------------------------------------------------------------
# Helper: decode uploaded bytes -> OpenCV image
# ---------------------------------------------------------------------------

def _decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG/PNG/BMP/...) into a BGR NumPy array."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image data — unsupported format or corrupt file.")
    return img


# ---------------------------------------------------------------------------
# Helper: run OCR synchronously (called inside thread pool)
# ---------------------------------------------------------------------------

def _run_ocr(img: np.ndarray) -> Dict[str, Any]:
    """
    Run PaddleOCR on a BGR NumPy image and return structured results.

    Returns a dict with:
        text          (str)  — full concatenated transcript
        words         (list) — per-word dicts {text, confidence, bbox}
        avg_confidence (float)
    """
    if _ocr_engine is None:
        raise RuntimeError("OCR engine is not available (startup failed).")

    t0 = time.perf_counter()
    results = _ocr_engine.predict(img)
    elapsed = time.perf_counter() - t0

    words: List[Dict[str, Any]] = []
    lines_text: List[str] = []

    result = results[0] if results else None
    if result and isinstance(result, dict) and "rec_texts" in result:
        for text, confidence, bbox in zip(
            result.get("rec_texts", []),
            result.get("rec_scores", []),
            result.get("rec_polys", []),
        ):
            if confidence is None or float(confidence) < CONF_THRESHOLD:
                continue
            words.append({
                "text": text,
                "confidence": round(float(confidence), 4),
                "bbox": [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in bbox],
            })
            lines_text.append(text)

    full_text = " ".join(lines_text)
    avg_conf = (
        float(np.mean([w["confidence"] for w in words])) if words else 0.0
    )

    return {
        "text": full_text,
        "words": words,
        "avg_confidence": round(avg_conf, 4),
        "inference_time_s": round(elapsed, 4),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness check")
async def health() -> JSONResponse:
    """Return server status and engine availability."""
    engine_ok = _ocr_engine is not None
    return JSONResponse(
        content={
            "status": "ok" if engine_ok else "degraded",
            "engine_ready": engine_ok,
        },
        status_code=200 if engine_ok else 503,
    )


@app.post("/ocr", summary="Recognise handwritten text in an uploaded image")
async def ocr_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept an image file upload and return OCR results.

    **Request**: multipart/form-data with field ``file`` containing the image.

    **Response** (200):
    ```json
    {
      "text": "full transcript string",
      "words": [
        {"text": "word", "confidence": 0.95, "bbox": [[x,y], ...]}
      ],
      "avg_confidence": 0.93,
      "inference_time_s": 0.12
    }
    ```
    """
    if _ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not available.")

    # Validate content type
    content_type = (file.content_type or "").lower()
    allowed = ("image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp", "")
    if content_type and content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {content_type}. "
                   "Send JPEG, PNG, BMP, TIFF, or WebP.",
        )

    # Read and decode
    try:
        raw_bytes = await file.read()
        img = _decode_image(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Error reading uploaded file")
        raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}")

    # Dispatch to thread pool
    import asyncio

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(_executor, _run_ocr, img)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("OCR inference error")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    logger.info(
        f"OCR: {len(result['words'])} words, "
        f"avg_conf={result['avg_confidence']:.3f}, "
        f"time={result['inference_time_s']}s"
    )
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Must be 1 when using a shared global OCR engine
        log_level="info",
        reload=False,
    )
