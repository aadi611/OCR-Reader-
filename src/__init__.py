"""
PaddleOCR Handwritten Text Recognition - Source Package
"""

from .preprocess import preprocess_handwriting, deskew
from .augment import get_handwriting_augmentation, augment_batch
from .evaluate import character_error_rate, word_error_rate, sequence_accuracy
from .postprocess import postprocess, apply_rules

__all__ = [
    "preprocess_handwriting",
    "deskew",
    "get_handwriting_augmentation",
    "augment_batch",
    "character_error_rate",
    "word_error_rate",
    "sequence_accuracy",
    "postprocess",
    "apply_rules",
]
