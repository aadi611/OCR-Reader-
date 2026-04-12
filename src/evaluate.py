"""
Evaluation metrics for handwritten text recognition.

Implements Character Error Rate (CER), Word Error Rate (WER), Sequence
Accuracy, bootstrap confidence intervals, and a high-level model evaluation
function that runs over a dataset using any PaddleOCR-compatible engine.
"""

import sys
import random
from typing import Any, Dict, List, Optional, Tuple

import editdistance
import numpy as np


# ---------------------------------------------------------------------------
# Low-level metric functions
# ---------------------------------------------------------------------------

def character_error_rate(
    predictions: List[str],
    ground_truths: List[str],
) -> float:
    """
    Compute mean Character Error Rate (CER) over a list of string pairs.

    CER = edit_distance(prediction, ground_truth) / len(ground_truth)

    Strings with empty ground truth are skipped to avoid division by zero.

    Args:
        predictions:   List of predicted strings.
        ground_truths: List of reference strings (same length).

    Returns:
        Mean CER as a float in [0, inf).  Returns 0.0 if the list is empty.
    """
    if not predictions:
        return 0.0
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length")

    rates = []
    for pred, gt in zip(predictions, ground_truths):
        if len(gt) == 0:
            continue
        rates.append(editdistance.eval(pred, gt) / len(gt))

    return float(np.mean(rates)) if rates else 0.0


def word_error_rate(
    predictions: List[str],
    ground_truths: List[str],
) -> float:
    """
    Compute mean Word Error Rate (WER) over a list of string pairs.

    WER = word_edit_distance(prediction, ground_truth) / n_words(ground_truth)

    Args:
        predictions:   List of predicted strings.
        ground_truths: List of reference strings.

    Returns:
        Mean WER as a float in [0, inf).
    """
    if not predictions:
        return 0.0
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length")

    rates = []
    for pred, gt in zip(predictions, ground_truths):
        pred_words = pred.split()
        gt_words = gt.split()
        if len(gt_words) == 0:
            continue
        dist = editdistance.eval(pred_words, gt_words)
        rates.append(dist / len(gt_words))

    return float(np.mean(rates)) if rates else 0.0


def sequence_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    ignore_case: bool = False,
    ignore_space: bool = False,
) -> float:
    """
    Compute exact sequence accuracy (fraction of perfect matches).

    Args:
        predictions:   List of predicted strings.
        ground_truths: List of reference strings.
        ignore_case:   If True, comparison is case-insensitive.
        ignore_space:  If True, leading/trailing spaces are stripped and
                       internal whitespace is normalised.

    Returns:
        Accuracy as a float in [0.0, 1.0].
    """
    if not predictions:
        return 0.0
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length")

    def normalize(s: str) -> str:
        if ignore_space:
            s = " ".join(s.split())
        if ignore_case:
            s = s.lower()
        return s

    correct = sum(normalize(p) == normalize(g) for p, g in zip(predictions, ground_truths))
    return correct / len(predictions)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_cer(
    preds: List[str],
    gts: List[str],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: Optional[int] = 42,
) -> Tuple[float, float, float]:
    """
    Estimate CER with bootstrap confidence interval.

    Resamples (with replacement) ``n_bootstrap`` times, computes CER for
    each resample, and returns the percentile-based CI.

    Args:
        preds:       List of predicted strings.
        gts:         List of reference strings.
        n_bootstrap: Number of bootstrap iterations.
        confidence:  Confidence level for the interval (e.g. 0.95 for 95%).
        random_seed: Seed for reproducibility; None for random.

    Returns:
        Tuple of (mean_cer, ci_low, ci_high).
    """
    if len(preds) != len(gts):
        raise ValueError("preds and gts must have the same length")

    rng = random.Random(random_seed)
    n = len(preds)
    indices = list(range(n))
    cer_samples: List[float] = []

    for _ in range(n_bootstrap):
        sample_idx = [rng.choice(indices) for _ in range(n)]
        sample_preds = [preds[i] for i in sample_idx]
        sample_gts = [gts[i] for i in sample_idx]
        cer_samples.append(character_error_rate(sample_preds, sample_gts))

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(cer_samples, 100 * alpha / 2))
    ci_high = float(np.percentile(cer_samples, 100 * (1 - alpha / 2)))
    mean_cer = float(np.mean(cer_samples))

    return mean_cer, ci_low, ci_high


# ---------------------------------------------------------------------------
# Stratified evaluation
# ---------------------------------------------------------------------------

def _stratify_by_length(
    preds: List[str],
    gts: List[str],
    bins: List[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Stratify evaluation by ground-truth string length.

    Args:
        preds:  Predictions.
        gts:    Ground truths.
        bins:   Length boundaries for strata.  Default: [0, 5, 10, 20, inf].

    Returns:
        Dict mapping stratum label -> {"predictions": ..., "ground_truths": ..., "n": ...}
    """
    if bins is None:
        bins = [0, 5, 10, 20, sys.maxsize]

    strata: Dict[str, Dict] = {}
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        label = f"len_{low}_{high if high < sys.maxsize else 'inf'}"
        mask = [low < len(gt) <= high for gt in gts]
        strata[label] = {
            "predictions": [p for p, m in zip(preds, mask) if m],
            "ground_truths": [g for g, m in zip(gts, mask) if m],
            "n": sum(mask),
        }

    return strata


def evaluate_model(
    ocr_engine: Any,
    test_data: List[Dict[str, Any]],
    preprocess_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run the OCR engine over a test dataset and compute aggregated metrics.

    ``test_data`` should be a list of dicts with at least:
        - ``"image_path"`` (str): path to the image file
        - ``"text"`` (str): ground-truth transcription

    The ``ocr_engine`` must be a callable that accepts a file path (str) and
    returns the recognised text (str) or a PaddleOCR-style result list.

    Args:
        ocr_engine:    A callable or PaddleOCR instance.
        test_data:     List of evaluation samples.
        preprocess_fn: Optional preprocessing function applied to each image
                       path before passing to the engine. Receives image path
                       (str) and returns a numpy array.

    Returns:
        Dict containing:
            - ``"cer"``         : mean Character Error Rate
            - ``"wer"``         : mean Word Error Rate
            - ``"seq_acc"``     : Sequence Accuracy
            - ``"n_samples"``   : number of evaluated samples
            - ``"cer_ci"``      : (mean, low, high) bootstrap 95% CI for CER
            - ``"stratified"``  : per-length-stratum metrics
    """
    import cv2

    predictions: List[str] = []
    ground_truths: List[str] = []

    for sample in test_data:
        image_path: str = sample["image_path"]
        gt_text: str = sample["text"]

        try:
            if preprocess_fn is not None:
                img = cv2.imread(image_path)
                if img is None:
                    raise FileNotFoundError(f"Cannot read: {image_path}")
                img = preprocess_fn(img)
                # Write preprocessed image to a temp file for engines that need a path
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                cv2.imwrite(tmp_path, img)
                raw = ocr_engine(tmp_path)
                import os
                os.unlink(tmp_path)
            else:
                raw = ocr_engine(image_path)

            # Normalise PaddleOCR output format
            pred_text = _extract_text_from_ocr_result(raw)
        except Exception as exc:
            print(f"[WARN] Failed on {image_path}: {exc}", file=sys.stderr)
            pred_text = ""

        predictions.append(pred_text)
        ground_truths.append(gt_text)

    cer = character_error_rate(predictions, ground_truths)
    wer = word_error_rate(predictions, ground_truths)
    seq_acc = sequence_accuracy(predictions, ground_truths)
    cer_ci = bootstrap_cer(predictions, ground_truths, n_bootstrap=1000)

    strata_raw = _stratify_by_length(predictions, ground_truths)
    stratified: Dict[str, Any] = {}
    for stratum_name, stratum_data in strata_raw.items():
        if stratum_data["n"] == 0:
            continue
        stratified[stratum_name] = {
            "n": stratum_data["n"],
            "cer": character_error_rate(
                stratum_data["predictions"], stratum_data["ground_truths"]
            ),
            "wer": word_error_rate(
                stratum_data["predictions"], stratum_data["ground_truths"]
            ),
            "seq_acc": sequence_accuracy(
                stratum_data["predictions"], stratum_data["ground_truths"]
            ),
        }

    return {
        "cer": cer,
        "wer": wer,
        "seq_acc": seq_acc,
        "n_samples": len(test_data),
        "cer_ci": cer_ci,
        "stratified": stratified,
    }


def _extract_text_from_ocr_result(result: Any) -> str:
    """
    Extract a single concatenated text string from various OCR output formats.

    Handles:
      - str: returned as-is
      - list of lists (PaddleOCR format): [[bbox, (text, score)], ...]
      - list of dicts with a "text" key
    """
    if isinstance(result, str):
        return result

    if isinstance(result, list):
        parts: List[str] = []
        for item in result:
            if isinstance(item, list):
                # PaddleOCR page-level: [[[bbox, (text, conf)], ...], ...]
                for line in item:
                    if isinstance(line, list) and len(line) == 2:
                        text_conf = line[1]
                        if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 1:
                            parts.append(str(text_conf[0]))
                    elif isinstance(line, dict) and "text" in line:
                        parts.append(str(line["text"]))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return " ".join(parts)

    return str(result)


# ---------------------------------------------------------------------------
# CLI / __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage demonstrating each metric function
    sample_preds = [
        "Hello world",
        "The quick brown fox",
        "handwriting recogniton",  # intentional typo
        "OpenCV is great",
    ]
    sample_gts = [
        "Hello world",
        "The quick brown fox",
        "handwriting recognition",
        "OpenCV is great",
    ]

    cer = character_error_rate(sample_preds, sample_gts)
    wer = word_error_rate(sample_preds, sample_gts)
    acc = sequence_accuracy(sample_preds, sample_gts)
    mean_cer, ci_low, ci_high = bootstrap_cer(sample_preds, sample_gts, n_bootstrap=1000)

    print("=" * 50)
    print("Evaluation Example")
    print("=" * 50)
    print(f"CER      : {cer:.4f} ({cer * 100:.2f}%)")
    print(f"WER      : {wer:.4f} ({wer * 100:.2f}%)")
    print(f"SeqAcc   : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"CER (bootstrap 95% CI): {mean_cer:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    # Per-sample breakdown
    print("\nPer-sample breakdown:")
    print(f"{'Prediction':<30} {'Ground Truth':<30} {'CER':>6}")
    print("-" * 70)
    for p, g in zip(sample_preds, sample_gts):
        sample_cer = editdistance.eval(p, g) / max(len(g), 1)
        print(f"{p:<30} {g:<30} {sample_cer:>6.4f}")
