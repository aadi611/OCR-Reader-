"""
Data augmentation pipeline for handwritten text recognition.

Provides albumentations-based augmentation for training images and a
synthetic data generator that renders text strings with custom fonts
using trdg (TextRecognitionDataGenerator).
"""

import os
import sys
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import albumentations as A
from albumentations.core.composition import Compose


def get_handwriting_augmentation(p: float = 0.5) -> Compose:
    """
    Build an albumentations augmentation pipeline tuned for handwriting.

    Applies geometric distortions, noise, blur, and quality degradations
    that mimic real-world variability in scanned or photographed handwriting.

    Args:
        p: Overall probability of applying each individual transform
           (used as the default where not overridden).

    Returns:
        albumentations.Compose pipeline ready for use.
    """
    return A.Compose(
        [
            # Geometric: small shifts, scale, and rotation
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                p=p,
            ),
            # Elastic deformation — simulates natural pen variation
            A.ElasticTransform(
                alpha=40,
                sigma=5,
                alpha_affine=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                p=p * 0.6,
            ),
            # Perspective warp — simulates camera angle variation
            A.Perspective(
                scale=(0.02, 0.06),
                pad_mode=cv2.BORDER_CONSTANT,
                pad_val=255,
                p=p * 0.4,
            ),
            # Gaussian noise — sensor/scan noise
            A.GaussNoise(
                var_limit=(5.0, 30.0),
                mean=0,
                p=p,
            ),
            # Gaussian blur — focus variation
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=p * 0.5,
            ),
            # Brightness and contrast — lighting variation
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=p,
            ),
            # Downscale — simulates low-resolution scanning then upscaling
            A.Downscale(
                scale_min=0.5,
                scale_max=0.9,
                interpolation=cv2.INTER_LINEAR,
                p=p * 0.3,
            ),
            # JPEG compression artifacts
            A.ImageCompression(
                quality_lower=60,
                quality_upper=95,
                p=p * 0.3,
            ),
            # Sharpening — over-sharpened scanner output
            A.Sharpen(
                alpha=(0.1, 0.3),
                lightness=(0.9, 1.1),
                p=p * 0.3,
            ),
            # Random ink dropout / occlusion (CoarseDropout)
            A.OneOf(
                [
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=8,
                        max_width=8,
                        min_holes=1,
                        min_height=2,
                        min_width=2,
                        fill_value=255,
                        p=1.0,
                    ),
                ],
                p=p * 0.2,
            ),
        ]
    )


def augment_batch(images: List[np.ndarray], augmentation: Optional[Compose] = None) -> List[np.ndarray]:
    """
    Apply augmentation to a list of images.

    Args:
        images: List of images as NumPy arrays (H x W or H x W x C).
        augmentation: albumentations Compose pipeline. If None, a default
                      pipeline is created via get_handwriting_augmentation().

    Returns:
        List of augmented images with the same length as the input.
    """
    if augmentation is None:
        augmentation = get_handwriting_augmentation()

    augmented = []
    for img in images:
        # albumentations expects uint8 HxWxC; handle grayscale
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_rgb = img.copy()

        result = augmentation(image=img_rgb)["image"]

        # Return in the same channel format as input
        if len(img.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        augmented.append(result)

    return augmented


def generate_synthetic(
    strings: List[str],
    font_path: Optional[str],
    output_dir: str,
    count: int = 1,
    image_height: int = 64,
) -> List[str]:
    """
    Generate synthetic handwriting-style text images using trdg.

    For each string in ``strings``, ``count`` images are generated and saved
    under ``output_dir``.  A label file ``labels.txt`` is also written to
    ``output_dir`` in PaddleOCR format (``<rel_path>\\t<text>``).

    Args:
        strings:     List of text strings to render.
        font_path:   Path to a TTF/OTF font file. If None, trdg uses its
                     bundled fonts.
        output_dir:  Directory where images and label file are saved.
        count:       Number of images to generate per string.
        image_height: Height in pixels of generated images.

    Returns:
        List of absolute paths to generated image files.
    """
    try:
        from trdg.generators import GeneratorFromStrings
    except ImportError as exc:
        raise ImportError(
            "trdg is required for synthetic data generation. "
            "Install it with: pip install trdg"
        ) from exc

    os.makedirs(output_dir, exist_ok=True)

    fonts = [font_path] if font_path and os.path.isfile(font_path) else []

    generator = GeneratorFromStrings(
        strings=strings,
        count=count,
        fonts=fonts if fonts else None,
        language="en",
        size=image_height,
        skewing_angle=3,
        random_skew=True,
        blur=1,
        random_blur=True,
        background_type=0,  # 0 = Gaussian noise background
        distorsion_type=3,  # 3 = random distortion
        distorsion_orientation=0,
        is_handwritten=False,  # set True if trdg version supports it
        width=-1,
        alignment=1,
        text_color="#000000",
        orientation=0,
        space_width=1.0,
        character_spacing=0,
        margins=(5, 5, 5, 5),
        fit=False,
    )

    saved_paths: List[str] = []
    label_lines: List[str] = []

    for idx, (img, label) in enumerate(generator):
        filename = f"synth_{idx:06d}.jpg"
        filepath = os.path.join(output_dir, filename)
        # trdg returns PIL images
        img_array = np.array(img)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img_array)
        saved_paths.append(filepath)
        label_lines.append(f"{filename}\t{label}")

    label_file = os.path.join(output_dir, "labels.txt")
    with open(label_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(label_lines) + "\n")

    print(f"Generated {len(saved_paths)} images -> {output_dir}")
    print(f"Labels file: {label_file}")
    return saved_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test augmentation pipeline")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output-dir", default="./aug_test", help="Output directory")
    parser.add_argument("--n", type=int, default=5, help="Number of augmented copies")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Cannot read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    augmented = augment_batch([img] * args.n)

    for i, aug_img in enumerate(augmented):
        out_path = os.path.join(args.output_dir, f"aug_{i:03d}.jpg")
        cv2.imwrite(out_path, aug_img)
        print(f"Saved: {out_path}")
