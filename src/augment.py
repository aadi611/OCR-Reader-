"""
handwriting_augmentation.py

Optimized data augmentation and synthetic-data generation pipeline
for handwritten text recognition.

Features:
    - Albumentations-based handwriting augmentation
    - Grayscale-aware processing
    - Configurable augmentation strength
    - Batch augmentation
    - Synthetic text generation with TRDG
    - Optional post-generation augmentation
    - PaddleOCR-compatible labels.txt
    - CLI for quick testing

Install:
    pip install opencv-python numpy albumentations trdg

Example:
    python handwriting_augmentation.py image.jpg --output-dir ./augmented --n 10

Synthetic generation:
    python handwriting_augmentation.py --synthetic \
        --output-dir ./synthetic \
        --text "Hello World" "Handwritten OCR" \
        --font ./fonts/handwriting.ttf \
        --count 10
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np
import albumentations as A


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for handwriting image augmentation."""

    probability: float = 0.5

    # Geometry
    shift_limit: float = 0.05
    scale_limit: float = 0.10
    rotate_limit: float = 5.0

    # Elastic deformation
    elastic_probability: float = 0.30
    elastic_alpha: float = 40.0
    elastic_sigma: float = 5.0

    # Perspective
    perspective_probability: float = 0.20
    perspective_scale: tuple[float, float] = (0.02, 0.06)

    # Noise / blur
    noise_probability: float = 0.50
    blur_probability: float = 0.25

    # Lighting
    brightness_probability: float = 0.50
    brightness_limit: float = 0.20
    contrast_limit: float = 0.20

    # Resolution degradation
    downscale_probability: float = 0.15
    downscale_range: tuple[float, float] = (0.5, 0.9)

    # Compression
    compression_probability: float = 0.15

    # Sharpen
    sharpen_probability: float = 0.15

    # Ink dropout
    dropout_probability: float = 0.10


# ============================================================================
# Augmentation pipeline
# ============================================================================


def build_augmentation(
    config: Optional[AugmentationConfig] = None,
) -> A.Compose:
    """
    Create an optimized Albumentations pipeline for handwriting OCR.

    The pipeline intentionally avoids excessive transformations. OCR models
    benefit from realistic variation, but aggressive augmentation can destroy
    character structure.

    Args:
        config: Optional augmentation configuration.

    Returns:
        Albumentations Compose object.
    """

    cfg = config or AugmentationConfig()

    p = float(np.clip(cfg.probability, 0.0, 1.0))

    transforms = [
        # ------------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------------

        A.ShiftScaleRotate(
            shift_limit=cfg.shift_limit,
            scale_limit=cfg.scale_limit,
            rotate_limit=cfg.rotate_limit,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            p=p,
        ),

        A.ElasticTransform(
            alpha=cfg.elastic_alpha,
            sigma=cfg.elastic_sigma,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            p=cfg.elastic_probability,
        ),

        A.Perspective(
            scale=cfg.perspective_scale,
            pad_mode=cv2.BORDER_CONSTANT,
            pad_val=255,
            p=cfg.perspective_probability,
        ),

        # ------------------------------------------------------------------
        # Image degradation
        # ------------------------------------------------------------------

        A.GaussNoise(
            std_range=(0.02, 0.08),
            mean_range=(0.0, 0.0),
            p=cfg.noise_probability,
        ),

        A.GaussianBlur(
            blur_limit=(3, 5),
            p=cfg.blur_probability,
        ),

        # ------------------------------------------------------------------
        # Lighting / contrast
        # ------------------------------------------------------------------

        A.RandomBrightnessContrast(
            brightness_limit=cfg.brightness_limit,
            contrast_limit=cfg.contrast_limit,
            p=cfg.brightness_probability,
        ),

        # ------------------------------------------------------------------
        # Resolution degradation
        # ------------------------------------------------------------------

        A.Downscale(
            scale_range=cfg.downscale_range,
            interpolation_pair={
                "downscale": cv2.INTER_AREA,
                "upscale": cv2.INTER_LINEAR,
            },
            p=cfg.downscale_probability,
        ),

        # ------------------------------------------------------------------
        # Compression
        # ------------------------------------------------------------------

        A.ImageCompression(
            quality_range=(60, 95),
            p=cfg.compression_probability,
        ),

        # ------------------------------------------------------------------
        # Sharpening
        # ------------------------------------------------------------------

        A.Sharpen(
            alpha=(0.1, 0.3),
            lightness=(0.9, 1.1),
            p=cfg.sharpen_probability,
        ),

        # ------------------------------------------------------------------
        # Small ink dropout
        # ------------------------------------------------------------------

        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(2, 8),
            hole_width_range=(2, 8),
            fill=255,
            p=cfg.dropout_probability,
        ),
    ]

    return A.Compose(transforms)


# ============================================================================
# Image utilities
# ============================================================================


def _prepare_image(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Prepare an image for Albumentations.

    Returns:
        processed_image:
        was_grayscale:
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image is empty.")

    was_grayscale = image.ndim == 2

    if was_grayscale:
        # Albumentations works efficiently with a normal image array.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    elif image.ndim == 3 and image.shape[2] in (3, 4):
        image = image.copy()

    else:
        raise ValueError(
            f"Unsupported image shape: {image.shape}. "
            "Expected HxW, HxWx3, or HxWx4."
        )

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image, was_grayscale


def _restore_image(
    image: np.ndarray,
    was_grayscale: bool,
) -> np.ndarray:
    """Restore the original channel format."""

    if was_grayscale:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


# ============================================================================
# Single image augmentation
# ============================================================================


def augment_image(
    image: np.ndarray,
    augmentation: Optional[A.Compose] = None,
) -> np.ndarray:
    """
    Augment a single image.

    Args:
        image: HxW grayscale or HxWx3 color image.
        augmentation: Optional pre-built augmentation pipeline.

    Returns:
        Augmented image with the same channel structure as input.
    """

    pipeline = augmentation or build_augmentation()

    prepared, was_grayscale = _prepare_image(image)

    result = pipeline(image=prepared)["image"]

    return _restore_image(result, was_grayscale)


# ============================================================================
# Batch augmentation
# ============================================================================


def augment_batch(
    images: Sequence[np.ndarray],
    augmentation: Optional[A.Compose] = None,
) -> List[np.ndarray]:
    """
    Augment a batch of images.

    The augmentation pipeline is created only once, which is important for
    training loops and large datasets.

    Args:
        images: Sequence of images.
        augmentation: Optional pre-built pipeline.

    Returns:
        List of augmented images.
    """

    if not images:
        return []

    pipeline = augmentation or build_augmentation()

    return [
        augment_image(image, pipeline)
        for image in images
    ]


# ============================================================================
# Synthetic data generation
# ============================================================================


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for TRDG synthetic generation."""

    image_height: int = 64
    count: int = 1

    skew_angle: int = 3
    blur: int = 1

    background_type: int = 0
    distortion_type: int = 3
    distortion_orientation: int = 0

    text_color: str = "#000000"

    margins: tuple[int, int, int, int] = (5, 5, 5, 5)

    # TRDG generates images using PIL.
    output_format: str = "jpg"


def _load_trdg():
    """Import TRDG lazily so normal augmentation doesn't require it."""

    try:
        from trdg.generators import GeneratorFromStrings
        return GeneratorFromStrings

    except ImportError as exc:
        raise ImportError(
            "TRDG is required for synthetic generation.\n"
            "Install it with:\n\n"
            "    pip install trdg"
        ) from exc


def generate_synthetic(
    strings: Sequence[str],
    output_dir: str | Path,
    font_path: Optional[str | Path] = None,
    config: Optional[SyntheticConfig] = None,
    augmentation: Optional[A.Compose] = None,
    augment_generated: bool = False,
) -> List[str]:
    """
    Generate synthetic handwriting/OCR images.

    Args:
        strings:
            Text strings to render.

        output_dir:
            Directory where generated images are saved.

        font_path:
            Optional TTF/OTF font.

        config:
            Synthetic generation configuration.

        augmentation:
            Optional Albumentations pipeline applied after generation.

        augment_generated:
            Whether to apply the augmentation pipeline to generated images.

    Returns:
        Absolute paths of generated image files.
    """

    if not strings:
        raise ValueError("At least one text string is required.")

    cfg = config or SyntheticConfig()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    GeneratorFromStrings = _load_trdg()

    # ----------------------------------------------------------------------
    # Validate font
    # ----------------------------------------------------------------------

    fonts: Optional[List[str]] = None

    if font_path is not None:
        font = Path(font_path)

        if not font.is_file():
            raise FileNotFoundError(
                f"Font file does not exist: {font}"
            )

        fonts = [str(font)]

    # ----------------------------------------------------------------------
    # TRDG generator
    # ----------------------------------------------------------------------

    generator = GeneratorFromStrings(
        strings=list(strings),
        count=cfg.count,

        fonts=fonts,

        language="en",
        size=cfg.image_height,

        skewing_angle=cfg.skew_angle,
        random_skew=True,

        blur=cfg.blur,
        random_blur=True,

        background_type=cfg.background_type,

        distorsion_type=cfg.distortion_type,
        distorsion_orientation=cfg.distortion_orientation,

        width=-1,
        alignment=1,

        text_color=cfg.text_color,

        orientation=0,

        space_width=1.0,
        character_spacing=0,

        margins=cfg.margins,

        fit=False,
    )

    if augment_generated and augmentation is None:
        augmentation = build_augmentation()

    saved_paths: List[str] = []
    labels: List[str] = []

    # ----------------------------------------------------------------------
    # Generate
    # ----------------------------------------------------------------------

    for index, (pil_image, label) in enumerate(generator):

        filename = f"synth_{index:06d}.{cfg.output_format}"
        filepath = output_path / filename

        image = np.asarray(pil_image)

        # PIL -> OpenCV
        if image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGBA2BGR,
                )
            elif image.shape[2] == 3:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR,
                )

        if augment_generated:
            image = augment_image(
                image,
                augmentation,
            )

        success = cv2.imwrite(
            str(filepath),
            image,
        )

        if not success:
            raise IOError(
                f"Failed to save generated image: {filepath}"
            )

        saved_paths.append(str(filepath.resolve()))
        labels.append(f"{filename}\t{label}")

    # ----------------------------------------------------------------------
    # PaddleOCR-style labels
    # ----------------------------------------------------------------------

    label_file = output_path / "labels.txt"

    label_file.write_text(
        "\n".join(labels) + "\n",
        encoding="utf-8",
    )

    print(
        f"Generated {len(saved_paths)} images -> "
        f"{output_path.resolve()}"
    )

    print(f"Labels: {label_file.resolve()}")

    return saved_paths


# ============================================================================
# Dataset augmentation helper
# ============================================================================


def augment_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    copies_per_image: int = 1,
    augmentation: Optional[A.Compose] = None,
) -> int:
    """
    Augment every supported image in a directory.

    Supported:
        .jpg
        .jpeg
        .png
        .bmp
        .tif
        .tiff

    Returns:
        Number of generated images.
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {input_path}"
        )

    if copies_per_image < 1:
        raise ValueError("copies_per_image must be >= 1.")

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline = augmentation or build_augmentation()

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    generated = 0

    for image_path in sorted(input_path.iterdir()):

        if image_path.suffix.lower() not in extensions:
            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            print(
                f"Warning: unable to read {image_path}",
                file=sys.stderr,
            )
            continue

        for copy_index in range(copies_per_image):

            augmented = augment_image(
                image,
                pipeline,
            )

            filename = (
                f"{image_path.stem}"
                f"_aug_{copy_index:03d}"
                f"{image_path.suffix}"
            )

            destination = output_path / filename

            if not cv2.imwrite(
                str(destination),
                augmented,
            ):
                raise IOError(
                    f"Failed to save: {destination}"
                )

            generated += 1

    print(
        f"Generated {generated} augmented images -> "
        f"{output_path.resolve()}"
    )

    return generated


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Handwriting OCR augmentation and "
            "synthetic-data generator."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ------------------------------------------------------------------
    # Single image
    # ------------------------------------------------------------------

    image_parser = subparsers.add_parser(
        "image",
        help="Augment a single image.",
    )

    image_parser.add_argument(
        "image",
        help="Input image path.",
    )

    image_parser.add_argument(
        "-o",
        "--output-dir",
        default="./aug_test",
        help="Output directory.",
    )

    image_parser.add_argument(
        "-n",
        "--copies",
        type=int,
        default=5,
        help="Number of augmented copies.",
    )

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------

    directory_parser = subparsers.add_parser(
        "directory",
        help="Augment all images in a directory.",
    )

    directory_parser.add_argument(
        "input_dir",
        help="Input image directory.",
    )

    directory_parser.add_argument(
        "-o",
        "--output-dir",
        default="./augmented",
        help="Output directory.",
    )

    directory_parser.add_argument(
        "-n",
        "--copies",
        type=int,
        default=1,
        help="Copies per input image.",
    )

    # ------------------------------------------------------------------
    # Synthetic
    # ------------------------------------------------------------------

    synthetic_parser = subparsers.add_parser(
        "synthetic",
        help="Generate synthetic OCR data.",
    )

    synthetic_parser.add_argument(
        "--text",
        nargs="+",
        required=True,
        help="Text strings to generate.",
    )

    synthetic_parser.add_argument(
        "-o",
        "--output-dir",
        default="./synthetic",
        help="Output directory.",
    )

    synthetic_parser.add_argument(
        "--font",
        default=None,
        help="TTF/OTF handwriting font.",
    )

    synthetic_parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Images generated per string.",
    )

    synthetic_parser.add_argument(
        "--height",
        type=int,
        default=64,
        help="Generated image height.",
    )

    synthetic_parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply Albumentations after TRDG generation.",
    )

    return parser


# ============================================================================
# CLI handlers
# ============================================================================


def main() -> None:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()

    # Build once and reuse.
    augmentation = build_augmentation()

    # ------------------------------------------------------------------
    # Single image
    # ------------------------------------------------------------------

    if args.command == "image":

        image = cv2.imread(
            args.image,
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {args.image}"
            )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index in range(args.copies):

            augmented = augment_image(
                image,
                augmentation,
            )

            output_path = (
                output_dir /
                f"aug_{index:03d}.jpg"
            )

            cv2.imwrite(
                str(output_path),
                augmented,
            )

            print(f"Saved: {output_path}")

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------

    elif args.command == "directory":

        augment_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            copies_per_image=args.copies,
            augmentation=augmentation,
        )

    # ------------------------------------------------------------------
    # Synthetic
    # ------------------------------------------------------------------

    elif args.command == "synthetic":

        config = SyntheticConfig(
            count=args.count,
            image_height=args.height,
        )

        generate_synthetic(
            strings=args.text,
            output_dir=args.output_dir,
            font_path=args.font,
            config=config,
            augmentation=augmentation,
            augment_generated=args.augment,
        )


if __name__ == "__main__":
    main()
