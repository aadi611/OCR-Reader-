"""
Language model integration for OCR post-decoding.

Provides:
  - CTC beam-search decoder with KenLM language model (via pyctcdecode).
  - GPT-2 based candidate reranking using perplexity scoring.
"""

import sys
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CTC decoder with KenLM
# ---------------------------------------------------------------------------

def build_ctc_decoder(
    vocab: List[str],
    kenlm_model_path: Optional[str] = None,
    alpha: float = 0.5,
    beta: float = 1.0,
    beam_width: int = 100,
):
    """
    Build a pyctcdecode BeamSearchDecoderCTC with optional KenLM scoring.

    The decoder accepts raw CTC log-probability matrices (shape: T x V) and
    returns the most likely transcript, optionally rescored with a n-gram LM.

    Args:
        vocab:            List of vocabulary tokens in the same order as CTC
                          output classes.  The blank token should be included
                          (pyctcdecode expects it as an empty string "").
        kenlm_model_path: Path to a KenLM binary (.klm) or ARPA (.arpa) file.
                          If None, decoding is performed without a language
                          model.
        alpha:            LM weight (higher = stronger language model influence).
        beta:             Word insertion bonus (higher = prefer longer sequences).
        beam_width:       Number of beams to maintain during search.

    Returns:
        pyctcdecode.BeamSearchDecoderCTC instance.

    Raises:
        ImportError: if pyctcdecode is not installed.
    """
    try:
        import pyctcdecode
        from pyctcdecode import build_ctcdecoder
    except ImportError as exc:
        raise ImportError(
            "pyctcdecode is required for CTC decoding with LM. "
            "Install it with: pip install pyctcdecode"
        ) from exc

    # pyctcdecode expects the blank token as an empty string ""
    # Ensure it is present at index 0 (CTC convention)
    if vocab and vocab[0] != "":
        vocab = [""] + vocab

    if kenlm_model_path is not None:
        decoder = build_ctcdecoder(
            labels=vocab,
            kenlm_model=kenlm_model_path,
            alpha=alpha,
            beta=beta,
        )
    else:
        decoder = build_ctcdecoder(labels=vocab)

    return decoder


def decode_with_lm(
    ctc_logits: np.ndarray,
    decoder,
    beam_width: int = 100,
    hotwords: Optional[List[str]] = None,
    hotword_weight: float = 10.0,
) -> str:
    """
    Decode CTC logits to text using a BeamSearchDecoderCTC.

    Args:
        ctc_logits:     NumPy array of shape (T, vocab_size) containing
                        raw logits or log-probabilities from the CTC head.
        decoder:        BeamSearchDecoderCTC instance from build_ctc_decoder().
        beam_width:     Beam width for search.
        hotwords:       Optional list of domain-specific words to boost.
        hotword_weight: Score bonus for hotwords.

    Returns:
        Decoded transcript string.
    """
    # pyctcdecode expects log probabilities; apply log-softmax if needed
    if ctc_logits.max() > 0:
        # Looks like raw logits — apply log-softmax
        from scipy.special import log_softmax
        logprobs = log_softmax(ctc_logits, axis=-1)
    else:
        logprobs = ctc_logits

    kwargs: Dict[str, Any] = {"beam_width": beam_width}
    if hotwords:
        kwargs["hotwords"] = hotwords
        kwargs["hotword_weight"] = hotword_weight

    text: str = decoder.decode(logprobs, **kwargs)
    return text


# ---------------------------------------------------------------------------
# GPT-2 reranking
# ---------------------------------------------------------------------------

def load_gpt2_lm(model_name: str = "gpt2"):
    """
    Load a GPT-2 language model and tokenizer for perplexity scoring.

    Args:
        model_name: Hugging Face model identifier (e.g. "gpt2", "gpt2-medium").

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        ImportError: if transformers or torch is not installed.
    """
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for GPT-2 reranking. "
            "Install them with: pip install transformers torch"
        ) from exc

    import torch

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, tokenizer


def _compute_perplexity(text: str, model, tokenizer, max_length: int = 512) -> float:
    """
    Compute GPT-2 perplexity for a given text string.

    Args:
        text:       Input text.
        model:      GPT2LMHeadModel.
        tokenizer:  GPT2TokenizerFast.
        max_length: Maximum number of tokens to consider.

    Returns:
        Perplexity as a float.  Lower is better.
    """
    import torch

    device = next(model.parameters()).device
    encodings = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )
    input_ids = encodings.input_ids.to(device)

    if input_ids.shape[1] < 2:
        # Too short to compute perplexity meaningfully
        return float("inf")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # cross-entropy loss == log-perplexity

    perplexity = math.exp(loss.item())
    return perplexity


def rerank_candidates(
    candidates: List[str],
    lm,
    tokenizer,
    top_k: int = 5,
    length_penalty: float = 0.0,
) -> List[Tuple[str, float]]:
    """
    Rerank OCR candidate strings using GPT-2 perplexity.

    Lower perplexity = better fluency.  Candidates are sorted ascending by
    perplexity (best first).

    Args:
        candidates:     List of candidate transcription strings.
        lm:             GPT-2 LMHeadModel instance.
        tokenizer:      GPT-2 tokenizer.
        top_k:          Return only the top-k ranked candidates.
        length_penalty: If > 0, adds a penalty proportional to candidate
                        length to bias towards shorter predictions.

    Returns:
        List of (candidate, perplexity) tuples sorted by ascending perplexity,
        truncated to top_k entries.
    """
    if not candidates:
        return []

    scored: List[Tuple[str, float]] = []
    for cand in candidates:
        if not cand.strip():
            scored.append((cand, float("inf")))
            continue
        ppl = _compute_perplexity(cand, lm, tokenizer)
        if length_penalty > 0.0:
            ppl += length_penalty * len(cand.split())
        scored.append((cand, ppl))

    # Sort ascending (lower perplexity = better)
    scored.sort(key=lambda x: x[1])
    return scored[:top_k]


# ---------------------------------------------------------------------------
# CLI / __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Language model reranking demo")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=[
            "The quick brown fox jumps over the lazy dog",
            "Tlie quick brown fox jumpz over tho lazy dog",
            "Quick the brown fox jumped over a lazy dog",
        ],
        help="Candidate strings to rerank",
    )
    parser.add_argument("--model", default="gpt2", help="GPT-2 model name")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k results to show")
    args = parser.parse_args()

    print(f"Loading GPT-2 model: {args.model}")
    try:
        lm, tokenizer = load_gpt2_lm(args.model)
        print("Model loaded. Reranking candidates...")
        ranked = rerank_candidates(args.candidates, lm, tokenizer, top_k=args.top_k)
        print(f"\nTop {args.top_k} candidates (sorted by perplexity):")
        print(f"{'Rank':<6} {'Perplexity':>12}  Candidate")
        print("-" * 70)
        for rank, (cand, ppl) in enumerate(ranked, 1):
            print(f"{rank:<6} {ppl:>12.2f}  {cand}")
    except ImportError as e:
        print(f"Cannot run demo: {e}", file=sys.stderr)
        sys.exit(1)
