"""
Post-processing pipeline for OCR output.

Provides:
  - Rule-based correction of common OCR substitution errors.
  - SymSpell-based spelling correction for word-level errors.
  - A unified postprocess() function that chains both steps.
"""

import re
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# SymSpell integration
# ---------------------------------------------------------------------------

def load_symspell(dict_path: str, max_edit_distance: int = 2, prefix_length: int = 7):
    """
    Load a SymSpell instance from a frequency dictionary file.

    The dictionary file must contain one entry per line in the format:
        word frequency
    (whitespace-separated).

    Args:
        dict_path:         Path to the frequency dictionary file.
        max_edit_distance: Maximum edit distance for lookup (default 2).
        prefix_length:     SymSpell prefix length parameter (default 7).

    Returns:
        Loaded symspellpy.SymSpell instance.

    Raises:
        ImportError: if symspellpy is not installed.
        FileNotFoundError: if the dictionary file does not exist.
        ValueError: if the dictionary could not be loaded.
    """
    try:
        from symspellpy import SymSpell, Verbosity  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "symspellpy is required for spelling correction. "
            "Install it with: pip install symspellpy"
        ) from exc

    import os

    if not os.path.isfile(dict_path):
        raise FileNotFoundError(f"SymSpell dictionary not found: {dict_path}")

    sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=prefix_length)
    loaded = sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    if not loaded:
        raise ValueError(f"Failed to load SymSpell dictionary from: {dict_path}")

    return sym_spell


def correct_ocr_output(
    text: str,
    sym_spell,
    max_edit_distance: int = 2,
) -> str:
    """
    Correct OCR output using SymSpell lookup.

    Each word in ``text`` is looked up independently.  If a better candidate
    is found within ``max_edit_distance`` edits, it replaces the original
    word.  Non-alphabetic tokens (numbers, punctuation) are preserved as-is.

    Args:
        text:              Input OCR string.
        sym_spell:         Loaded symspellpy.SymSpell instance.
        max_edit_distance: Maximum edit distance for correction.

    Returns:
        Corrected string.
    """
    from symspellpy import Verbosity

    if not text.strip():
        return text

    tokens = text.split()
    corrected_tokens: list = []

    for token in tokens:
        # Strip surrounding punctuation for lookup
        stripped, prefix, suffix = _strip_punctuation(token)

        if stripped and stripped.isalpha():
            suggestions = sym_spell.lookup(
                stripped,
                Verbosity.CLOSEST,
                max_edit_distance=max_edit_distance,
                include_unknown=True,
            )
            if suggestions:
                best = suggestions[0].term
                # Preserve original capitalisation style
                best = _match_case(stripped, best)
                corrected_tokens.append(prefix + best + suffix)
            else:
                corrected_tokens.append(token)
        else:
            corrected_tokens.append(token)

    return " ".join(corrected_tokens)


def _strip_punctuation(token: str):
    """
    Strip leading/trailing punctuation from a token.

    Returns:
        (core_word, prefix_punct, suffix_punct)
    """
    left = len(token) - len(token.lstrip(".,!?;:\"'()[]{}"))
    right = len(token) - len(token.rstrip(".,!?;:\"'()[]{}"))
    prefix = token[:left]
    suffix = token[len(token) - right:] if right else ""
    core = token[left: len(token) - right if right else None]
    return core, prefix, suffix


def _match_case(original: str, corrected: str) -> str:
    """Apply the capitalisation style of ``original`` to ``corrected``."""
    if not corrected:
        return corrected
    if original.isupper():
        return corrected.upper()
    if original.islower():
        return corrected.lower()
    if original and original[0].isupper():
        return corrected[0].upper() + corrected[1:].lower()
    return corrected


# ---------------------------------------------------------------------------
# Rule-based OCR error correction
# ---------------------------------------------------------------------------

# Common single-character substitution rules (applied before spell check)
# Ordered so that more specific patterns are applied before generic ones.
_CHAR_RULES = [
    # "rn" often misread as "m"
    (re.compile(r"rn"), "m"),
    # "cl" often misread as "d"
    (re.compile(r"cl"), "d"),
    # "vv" often misread as "w"
    (re.compile(r"vv"), "w"),
    # "1" (digit one) misread as "l" (lowercase L) inside all-alpha words
    (re.compile(r"(?<=[a-zA-Z])1(?=[a-zA-Z])"), "l"),
    # "0" (zero) misread as "O" (capital oh) inside all-alpha words
    (re.compile(r"(?<=[a-zA-Z])0(?=[a-zA-Z])"), "O"),
    # "5" misread as "S" at the start of a capitalised word
    (re.compile(r"(?<=\b)5(?=[a-z])"), "S"),
    # "I" (capital i) misread as "l" (ell) in all-lowercase context
    (re.compile(r"(?<=[a-z])I(?=[a-z])"), "l"),
    # Isolated "l" at word boundary that should be "I" (pronoun)
    (re.compile(r"\bl\b"), "I"),
]

# Whitespace normalisation
_MULTI_SPACE = re.compile(r" {2,}")


def apply_rules(text: str) -> str:
    """
    Apply regex-based common OCR substitution rules.

    Corrects frequent character confusions that arise from imperfect OCR
    models, such as:
      - "rn" -> "m"
      - "cl" -> "d"
      - "vv" -> "w"
      - digit "1" inside words -> letter "l"
      - digit "0" inside words -> letter "O"
      - Isolated "l" -> pronoun "I"

    Args:
        text: Raw OCR output string.

    Returns:
        String with rule-based substitutions applied.
    """
    for pattern, replacement in _CHAR_RULES:
        text = pattern.sub(replacement, text)
    # Normalise multiple spaces introduced by substitutions
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Unified pipeline
# ---------------------------------------------------------------------------

def postprocess(
    text: str,
    sym_spell=None,
    max_edit_distance: int = 2,
) -> str:
    """
    Full post-processing pipeline: rule correction then optional spell check.

    Args:
        text:              Raw OCR output string.
        sym_spell:         Optional loaded symspellpy.SymSpell instance.
                           If None, only rule-based correction is applied.
        max_edit_distance: Maximum edit distance for SymSpell correction.

    Returns:
        Post-processed string.
    """
    # Step 1: rule-based substitutions
    text = apply_rules(text)

    # Step 2: SymSpell word-level correction (optional)
    if sym_spell is not None:
        text = correct_ocr_output(text, sym_spell, max_edit_distance=max_edit_distance)

    return text


# ---------------------------------------------------------------------------
# CLI / __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test OCR post-processing")
    parser.add_argument("text", nargs="?", help="Text to post-process (or reads from stdin)")
    parser.add_argument("--dict", default=None, help="Path to SymSpell frequency dictionary")
    parser.add_argument("--max-edit", type=int, default=2, help="Max edit distance for SymSpell")
    args = parser.parse_args()

    if args.text:
        raw = args.text
    else:
        print("Enter text to post-process (Ctrl+D to finish):")
        raw = sys.stdin.read().strip()

    sym_spell = None
    if args.dict:
        try:
            print(f"Loading SymSpell dictionary: {args.dict}")
            sym_spell = load_symspell(args.dict, max_edit_distance=args.max_edit)
            print("  Dictionary loaded successfully.")
        except Exception as exc:
            print(f"  Warning: could not load dictionary: {exc}", file=sys.stderr)

    # Show rule-by-rule demonstration
    print("\nRule-based corrections:")
    after_rules = apply_rules(raw)
    if after_rules != raw:
        print(f"  Input : {raw!r}")
        print(f"  Output: {after_rules!r}")
    else:
        print("  No rule-based changes.")

    result = postprocess(raw, sym_spell=sym_spell, max_edit_distance=args.max_edit)

    print("\nFinal result:")
    print(f"  Input : {raw!r}")
    print(f"  Output: {result!r}")

    # Built-in test cases
    test_cases = [
        ("l went to the store", "I went to the store"),
        ("The cat rnan away", "The cat man away"),
        ("He clid it", "He did it"),
        ("vvater bottle", "water bottle"),
        ("t0tal cost", "tOtal cost"),
    ]
    print("\nBuilt-in test cases (rule-based only):")
    print(f"{'Input':<30} {'Expected':<30} {'Got':<30} {'Pass':>5}")
    print("-" * 100)
    for inp, expected in test_cases:
        got = apply_rules(inp)
        passed = "YES" if got == expected else "NO "
        print(f"{inp:<30} {expected:<30} {got:<30} {passed:>5}")
