"""
Command-line inference for handwritten text recognition (PaddleOCR).

Usage:
    python infer.py --image path/to/image.jpg
    python infer.py --image imgs/ --save-output out/            # batch a folder
    python infer.py --image "scans/*.png" --json results.json   # batch a glob
    python infer.py --image img.jpg --det-model ./output/det_handwriting/best_accuracy
    python infer.py --image img.jpg --no-gpu --conf-threshold 0.4 --save-output result.jpg

Only --image is required. It accepts a file, a directory, or a glob and may be
repeated; the (expensive) OCR engine is built once and reused for every image.
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np

# OpenCV spawns its own thread pool; leave a couple of cores for the inference
# runtime so the two don't fight over the machine.
cv2.setUseOptimized(True)
_CPU = os.cpu_count() or 4
cv2.setNumThreads(max(1, min(_CPU - 1, 8)))

IMAGE_EXTS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Word:
    """One recognised text region. Bounds are computed once, at parse time."""

    text: str
    confidence: float
    poly: np.ndarray  # (N, 2) float32, in ORIGINAL image coordinates
    x0: float
    y0: float
    x1: float
    y1: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": np.round(self.poly, 1).tolist(),
        }


@dataclass(slots=True)
class Prepared:
    """An image loaded, optionally preprocessed and resized, ready for OCR."""

    path: str
    original: np.ndarray
    model_input: np.ndarray
    scale: float  # model_input coords * (1 / scale) -> original coords
    error: str | None = None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Handwritten text recognition inference with PaddleOCR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        action="append",
        metavar="PATH",
        help="Image file, directory, or glob. May be given more than once.",
    )
    parser.add_argument("--det-model", default=None,
                        help="Detection model directory. Uses PP-OCRv4 default if unset.")
    parser.add_argument("--rec-model", default=None,
                        help="Recognition model directory. Uses PP-OCRv4 default if unset.")
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=None,
                        help="Force GPU/CPU. Default: auto-detect CUDA.")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                        help="Drop results below this confidence.")
    parser.add_argument("--rec-char-dict", default="./dict/en_dict.txt",
                        help="Recognition character dictionary (only used if it exists).")
    parser.add_argument("--save-output", default=None, metavar="PATH",
                        help="Write annotated image(s) here. A directory when batching.")
    parser.add_argument("--json", dest="json_path", default=None, metavar="PATH",
                        help="Write structured results to this JSON file.")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip the deskew + adaptive-threshold pipeline.")
    parser.add_argument("--max-side", type=int, default=1600, metavar="PX",
                        help="Downscale so the longest side is at most this. 0 disables.")
    parser.add_argument("--cpu-threads", type=int, default=max(1, min(_CPU, 8)),
                        help="Inference threads when running on CPU.")
    parser.add_argument("--warmup", action="store_true",
                        help="Run one throwaway pass so timings exclude lazy init.")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Print only the recognised text, one line per image.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------
def collect_images(patterns: Iterable[str]) -> list[str]:
    """Expand files, directories and globs into a de-duplicated, sorted list."""
    found: dict[str, None] = {}  # ordered set
    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            matches = sorted(
                str(f) for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS
            )
        elif p.is_file():
            matches = [str(p)]
        else:
            matches = sorted(
                m for m in globlib.glob(pattern, recursive=True)
                if Path(m).suffix.lower() in IMAGE_EXTS and os.path.isfile(m)
            )
        if not matches:
            print(f"[WARN] no images matched: {pattern}", file=sys.stderr)
        found.update(dict.fromkeys(matches))
    return list(found)


# ---------------------------------------------------------------------------
# Loading / preprocessing (runs on worker threads, overlapped with inference)
# ---------------------------------------------------------------------------
_preprocess_fn: Any = None          # resolved once
_preprocess_checked = False


def _get_preprocess():
    """Import src.preprocess exactly once instead of on every image."""
    global _preprocess_fn, _preprocess_checked
    if not _preprocess_checked:
        _preprocess_checked = True
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from src.preprocess import preprocess_handwriting
            _preprocess_fn = preprocess_handwriting
        except ImportError:
            print("[WARN] src.preprocess not available; skipping preprocessing.",
                  file=sys.stderr)
    return _preprocess_fn


def imread_unicode(path: str) -> np.ndarray | None:
    """cv2.imread that also works with non-ASCII paths."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except OSError:
        return None


def prepare(path: str, do_preprocess: bool, max_side: int) -> Prepared:
    img = imread_unicode(path)
    if img is None:
        empty = np.zeros((1, 1, 3), np.uint8)
        return Prepared(path, empty, empty, 1.0, error="cannot decode image")

    work = img
    scale = 1.0
    if max_side > 0:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > max_side:
            scale = max_side / longest
            # INTER_AREA is both faster and cleaner for downscaling.
            work = cv2.resize(img, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)

    if do_preprocess:
        fn = _get_preprocess()
        if fn is not None:
            work = fn(work)
            if work.ndim == 2:
                work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)

    return Prepared(path, img, np.ascontiguousarray(work), scale)


def iter_prepared(
    paths: Sequence[str], do_preprocess: bool, max_side: int, prefetch: int = 2
) -> Iterator[Prepared]:
    """Yield prepared images, decoding the next ones while OCR runs on this one."""
    if len(paths) == 1:
        yield prepare(paths[0], do_preprocess, max_side)
        return

    with ThreadPoolExecutor(max_workers=min(4, prefetch + 1)) as pool:
        it = iter(paths)
        queue = deque(
            pool.submit(prepare, p, do_preprocess, max_side)
            for p in islice(it, prefetch + 1)
        )
        while queue:
            fut = queue.popleft()
            nxt = next(it, None)
            if nxt is not None:
                queue.append(pool.submit(prepare, nxt, do_preprocess, max_side))
            yield fut.result()


# ---------------------------------------------------------------------------
# OCR engine
# ---------------------------------------------------------------------------
def detect_gpu() -> bool:
    try:
        import paddle  # noqa: PLC0415
        return bool(paddle.device.cuda.device_count())
    except Exception:
        return False


def load_ocr_engine(args: argparse.Namespace):
    """Build a PaddleOCR instance. Thread/env knobs are set before the import."""
    use_gpu = detect_gpu() if args.use_gpu is None else args.use_gpu
    if not use_gpu:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
        os.environ.setdefault("FLAGS_use_mkldnn", "1")

    try:
        from paddleocr import PaddleOCR  # noqa: PLC0415
    except ImportError:
        sys.exit("Error: paddleocr is not installed. Run: pip install paddleocr")

    kwargs: dict[str, Any] = {
        "use_textline_orientation": True,
        "lang": "en",
        "device": "gpu" if use_gpu else "cpu",
        "text_det_thresh": 0.3,
        "text_det_box_thresh": 0.5,
        "text_det_unclip_ratio": 2.0,
    }
    if args.det_model:
        kwargs["text_detection_model_dir"] = args.det_model
    if args.rec_model:
        kwargs["text_recognition_model_dir"] = args.rec_model
    if args.rec_char_dict and os.path.isfile(args.rec_char_dict):
        kwargs["character_dict_path"] = args.rec_char_dict
    if not use_gpu:
        kwargs["cpu_threads"] = args.cpu_threads
        kwargs["enable_mkldnn"] = True

    print(f"Loading PaddleOCR engine (device={kwargs['device']})...", file=sys.stderr)
    try:
        engine = PaddleOCR(**kwargs)
    except TypeError:
        # Older/newer builds reject some optional keys; retry with the core set.
        for key in ("cpu_threads", "enable_mkldnn", "character_dict_path"):
            kwargs.pop(key, None)
        engine = PaddleOCR(**kwargs)
    print("Engine loaded.", file=sys.stderr)

    if args.warmup:
        try:
            engine.predict(np.full((320, 320, 3), 255, np.uint8))
        except Exception:
            pass
    return engine


def predict(engine, image: np.ndarray):
    """Call whichever inference entry point this PaddleOCR version exposes."""
    fn = getattr(engine, "predict", None) or engine.ocr
    return fn(image)


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------
def _make_word(text: str, conf: float, poly: Any, inv_scale: float) -> Word:
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if inv_scale != 1.0:
        pts *= inv_scale
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return Word(text, conf, pts, float(x0), float(y0), float(x1), float(y1))


def parse_results(raw: Any, conf_threshold: float, scale: float = 1.0) -> list[Word]:
    """Normalise PaddleOCR 3.x and legacy outputs into a list of Words."""
    if not raw:
        return []
    inv_scale = 1.0 / scale if scale else 1.0
    result = raw[0] if isinstance(raw, (list, tuple)) else raw
    words: list[Word] = []
    try:
        if isinstance(result, dict) and "rec_texts" in result:
            texts = result.get("rec_texts") or ()
            scores = result.get("rec_scores") or ()
            polys = result.get("rec_polys")
            if polys is None:
                polys = result.get("dt_polys") or ()
            words = [
                _make_word(t, c, p, inv_scale)
                for t, c, p in zip(texts, scores, polys)
                if c is not None and c >= conf_threshold
            ]
        else:
            for line in (result or ()):
                if not line:
                    continue
                bbox, (text, conf) = line
                conf = float(conf)
                if conf >= conf_threshold:
                    words.append(_make_word(text, conf, bbox, inv_scale))
    except Exception as exc:
        print(f"[WARN] could not parse results: {exc}", file=sys.stderr)
    return words


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_report(path: str, words: Sequence[Word], elapsed: float) -> str:
    """Build the whole table as one string — one write beats hundreds."""
    if not words:
        return f"\n{path}: no text detected above confidence threshold."

    avg = sum(w.confidence for w in words) / len(words)
    out = [
        "\n" + "=" * 72,
        f"  RECOGNISED TEXT: {' '.join(w.text for w in words)}",
        "=" * 72,
        f"  Source         : {path}",
        f"  Words detected : {len(words)}",
        f"  Avg confidence : {avg:.4f} ({avg * 100:.1f}%)",
        f"  Inference time : {elapsed:.4f}s",
        "-" * 72,
        f"  {'#':<4} {'Text':<30} {'Conf':>6}  BBox (top-left x,y  w,h)",
        "-" * 72,
    ]
    out.extend(
        f"  {i:<4} {w.text:<30} {w.confidence:>6.4f}  "
        f"({w.x0:.0f}, {w.y0:.0f})  {w.x1 - w.x0:.0f}x{w.y1 - w.y0:.0f}"
        for i, w in enumerate(words, 1)
    )
    out.append("=" * 72)
    return "\n".join(out)


def save_annotated(image: np.ndarray, words: Sequence[Word], output_path: str) -> None:
    """Draw all boxes in a single polylines call, then the labels."""
    annotated = image.copy()
    if words:
        polys = [w.poly.round().astype(np.int32).reshape(-1, 1, 2) for w in words]
        cv2.polylines(annotated, polys, True, (0, 255, 0), 2, cv2.LINE_8)
        for w in words:
            cv2.putText(
                annotated,
                f"{w.text} ({w.confidence:.2f})",
                (max(int(w.x0), 0), max(int(w.y0) - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
            )

    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    if cv2.imwrite(output_path, annotated):
        print(f"Annotated image saved: {output_path} "
              f"({os.path.getsize(output_path) / 1024:.1f} KB)", file=sys.stderr)
    else:
        print(f"[WARN] failed to save: {output_path}", file=sys.stderr)


def resolve_output_path(save_output: str, src: str, batch: bool) -> str:
    if not batch and not (save_output.endswith(("/", os.sep)) or os.path.isdir(save_output)):
        return save_output
    stem = Path(src).stem
    return str(Path(save_output) / f"{stem}_annotated.jpg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    paths = collect_images(args.image)
    if not paths:
        print("Error: no input images found.", file=sys.stderr)
        return 1

    engine = load_ocr_engine(args)          # built once, reused for every image
    batch = len(paths) > 1
    payload: list[dict[str, Any]] = []
    total = 0.0

    for item in iter_prepared(paths, not args.no_preprocess, args.max_side):
        if item.error:
            print(f"[WARN] {item.path}: {item.error}", file=sys.stderr)
            continue

        t0 = time.perf_counter()
        raw = predict(engine, item.model_input)
        elapsed = time.perf_counter() - t0
        total += elapsed

        words = parse_results(raw, args.conf_threshold, item.scale)

        if args.quiet:
            print(f"{item.path}\t{' '.join(w.text for w in words)}")
        else:
            print(format_report(item.path, words, elapsed))

        if args.save_output:
            save_annotated(item.original, words,
                           resolve_output_path(args.save_output, item.path, batch))

        if args.json_path:
            payload.append({
                "image": item.path,
                "elapsed_s": round(elapsed, 4),
                "words": [w.as_dict() for w in words],
                "text": " ".join(w.text for w in words),
            })

    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"JSON written: {args.json_path}", file=sys.stderr)

    if batch:
        print(f"\n{len(paths)} images | total inference {total:.3f}s "
              f"| {total / len(paths):.3f}s per image", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
