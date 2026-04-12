"""
Command-line inference script for handwritten text recognition.

Usage:
    python infer.py --image path/to/image.jpg
    python infer.py --image img.jpg --det-model ./output/det_handwriting/best_accuracy
    python infer.py --image img.jpg --rec-model ./output/rec_handwriting_svtr/best_accuracy
    python infer.py --image img.jpg --no-gpu --conf-threshold 0.4 --save-output result.jpg

All arguments are optional except --image.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Handwritten text recognition inference with PaddleOCR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--det-model",
        default=None,
        help="Directory of the detection model. Uses PP-OCRv4 default if not set.",
    )
    parser.add_argument(
        "--rec-model",
        default=None,
        help="Directory of the recognition model. Uses PP-OCRv4 default if not set.",
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU for inference (--use-gpu / --no-gpu).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence threshold; results below this are discarded.",
    )
    parser.add_argument(
        "--rec-char-dict",
        default="./dict/en_dict.txt",
        help="Path to the recognition character dictionary file.",
    )
    parser.add_argument(
        "--save-output",
        default=None,
        metavar="PATH",
        help="If specified, save an annotated output image to this path.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        default=False,
        help="Skip the preprocessing pipeline (deskew + adaptive threshold).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing (imported from src)
# ---------------------------------------------------------------------------

def try_preprocess(img: np.ndarray) -> np.ndarray:
    """Apply preprocessing if available; fall back to the original image."""
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from src.preprocess import preprocess_handwriting

        preprocessed = preprocess_handwriting(img)
        # Convert grayscale back to BGR for PaddleOCR
        if len(preprocessed.shape) == 2:
            preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        return preprocessed
    except ImportError:
        print("[WARN] src.preprocess not available; skipping preprocessing.")
        return img


# ---------------------------------------------------------------------------
# OCR engine loader
# ---------------------------------------------------------------------------

def load_ocr_engine(args: argparse.Namespace):
    """Load and configure a PaddleOCR instance."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        print(
            "Error: paddleocr is not installed. Run: pip install paddleocr",
            file=sys.stderr,
        )
        sys.exit(1)

    device = "gpu" if args.use_gpu else "cpu"
    kwargs: Dict[str, Any] = dict(
        use_textline_orientation=True,
        lang="en",
        device=device,
        text_det_thresh=0.3,
        text_det_box_thresh=0.5,
        text_det_unclip_ratio=2.0,
    )
    if args.det_model:
        kwargs["text_detection_model_dir"] = args.det_model
    if args.rec_model:
        kwargs["text_recognition_model_dir"] = args.rec_model

    print(f"Loading PaddleOCR engine (device={device})...")
    engine = PaddleOCR(**kwargs)
    print("Engine loaded.")
    return engine


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_results(
    raw_results: Any,
    conf_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Parse PaddleOCR results into a clean list of word records.

    Each record contains:
        text       (str)
        confidence (float)
        bbox       (list of [x, y] corner points)
    """
    words: List[Dict[str, Any]] = []
    if not raw_results:
        return words

    try:
        result = raw_results[0] if isinstance(raw_results, list) else raw_results

        # PaddleOCR 3.x: OCRResult dict with rec_texts/rec_scores/rec_polys
        if isinstance(result, dict) and "rec_texts" in result:
            texts  = result.get("rec_texts", [])
            scores = result.get("rec_scores", [])
            polys  = result.get("rec_polys", [])
            for text, confidence, bbox in zip(texts, scores, polys):
                if confidence is None or float(confidence) < conf_threshold:
                    continue
                words.append({
                    "text": text,
                    "confidence": round(float(confidence), 4),
                    "bbox": [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in bbox],
                })
        else:
            # Legacy API: nested list [[bbox, (text, conf)], ...]
            src = result if result is not None else []
            for line in (src or []):
                if line is None:
                    continue
                bbox, (text, confidence) = line
                if float(confidence) < conf_threshold:
                    continue
                words.append({
                    "text": text,
                    "confidence": round(float(confidence), 4),
                    "bbox": [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in bbox],
                })
    except Exception as exc:
        print(f"[WARN] Could not parse results: {exc}", file=sys.stderr)
    return words


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_results_table(words: List[Dict[str, Any]], elapsed: float) -> None:
    """Print OCR results in a readable table format."""
    if not words:
        print("\nNo text detected above confidence threshold.")
        return

    full_text = " ".join(w["text"] for w in words)
    avg_conf = sum(w["confidence"] for w in words) / len(words)

    print("\n" + "=" * 72)
    print(f"  RECOGNISED TEXT: {full_text}")
    print("=" * 72)
    print(f"  Words detected : {len(words)}")
    print(f"  Avg confidence : {avg_conf:.4f} ({avg_conf * 100:.1f}%)")
    print(f"  Inference time : {elapsed:.4f}s")
    print("-" * 72)
    print(f"  {'#':<4} {'Text':<30} {'Conf':>6}  {'BBox (top-left x,y  w,h)'}")
    print("-" * 72)

    for idx, w in enumerate(words, 1):
        pts = w["bbox"]
        x_min = min(p[0] for p in pts)
        y_min = min(p[1] for p in pts)
        x_max = max(p[0] for p in pts)
        y_max = max(p[1] for p in pts)
        w_px = x_max - x_min
        h_px = y_max - y_min
        bbox_str = f"({x_min:.0f}, {y_min:.0f})  {w_px:.0f}x{h_px:.0f}"
        print(f"  {idx:<4} {w['text']:<30} {w['confidence']:>6.4f}  {bbox_str}")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Annotated output image
# ---------------------------------------------------------------------------

def save_annotated_image(
    image: np.ndarray,
    words: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Draw bounding boxes and text labels on the image and save it.

    Args:
        image:       Original BGR image.
        words:       List of word records from parse_results().
        output_path: Destination file path.
    """
    annotated = image.copy()

    for word in words:
        pts = np.array(word["bbox"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Label above the bounding box
        x, y = int(word["bbox"][0][0]), int(word["bbox"][0][1]) - 5
        label = f"{word['text']} ({word['confidence']:.2f})"
        cv2.putText(
            annotated,
            label,
            (max(x, 0), max(y, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    success = cv2.imwrite(output_path, annotated)
    if success:
        size_kb = os.path.getsize(output_path) / 1024
        print(f"\nAnnotated image saved: {output_path} ({size_kb:.1f} KB)")
    else:
        print(f"\n[WARN] Failed to save annotated image to: {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate input
    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: cannot decode image: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Image: {args.image}  ({img.shape[1]}x{img.shape[0]} px)")

    # Preprocessing
    if not args.no_preprocess:
        print("Applying preprocessing pipeline...")
        img_input = try_preprocess(img)
    else:
        img_input = img

    # Load engine
    engine = load_ocr_engine(args)

    # Run inference
    print("Running OCR inference...")
    t0 = time.perf_counter()
    raw_results = engine.predict(img_input)
    elapsed = time.perf_counter() - t0

    # Parse and display
    words = parse_results(raw_results, args.conf_threshold)
    print_results_table(words, elapsed)

    # Save annotated image if requested
    if args.save_output:
        save_annotated_image(img, words, args.save_output)


if __name__ == "__main__":
    main()
