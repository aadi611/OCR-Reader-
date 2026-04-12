"""
Image preprocessing pipeline for handwritten text recognition.

Provides deskewing via Hough line transform, adaptive thresholding,
morphological operations, and a full end-to-end preprocessing pipeline.
"""

import sys
import os
import math

import cv2
import numpy as np
from scipy import ndimage


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Deskew an image using Hough line transform to detect dominant text angle.

    Detects lines in the image via the probabilistic Hough transform, computes
    the median rotation angle, and rotates the image to straighten text.

    Args:
        image: Input grayscale or BGR image as a NumPy array.

    Returns:
        Deskewed image as a NumPy array of the same dtype.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Threshold to binary for line detection
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect edges for Hough transform
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return image

    # Compute angle for each detected line
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Filter angles close to horizontal (text lines)
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return image

    # Use median to robustly estimate skew angle
    median_angle = float(np.median(angles))

    # Only rotate if skew is significant (> 0.5 degrees)
    if abs(median_angle) < 0.5:
        return image

    # Rotate image to correct skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    # Use white background for padding
    if len(image.shape) == 3:
        border_value = (255, 255, 255)
    else:
        border_value = 255

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return rotated


def preprocess_handwriting(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline for handwritten text images.

    Steps:
      1. Convert to grayscale
      2. Deskew using Hough line transform
      3. Adaptive thresholding (Gaussian weighted)
      4. Morphological closing to connect broken strokes
      5. Non-local means denoising

    Args:
        image: Input image as a NumPy array (BGR or grayscale).

    Returns:
        Preprocessed binary image as a uint8 NumPy array.
    """
    # Step 1: Grayscale conversion
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 2: Deskew
    gray = deskew(gray)

    # Step 3: Adaptive thresholding — Gaussian method handles uneven illumination
    # blockSize must be odd; 11 works well for typical handwriting scans
    thresh = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    # Step 4: Morphological closing — fills small gaps in characters
    # Kernel sized for typical pen stroke width at ~150–300 dpi
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Step 5: Non-local means denoising — removes salt-and-pepper artifacts
    # h=10 is a conservative denoising strength suitable for binary-like images
    denoised = cv2.fastNlMeansDenoising(closed, h=10, templateWindowSize=7, searchWindowSize=21)

    return denoised


def load_image(path: str) -> np.ndarray:
    """Load an image from disk, raising if not found."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_image(image: np.ndarray, path: str) -> None:
    """Save a NumPy image array to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test handwriting preprocessing pipeline")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save preprocessed image (default: show with cv2.imshow)",
    )
    args = parser.parse_args()

    print(f"Loading image: {args.image}")
    img = load_image(args.image)
    print(f"  Original shape: {img.shape}, dtype: {img.dtype}")

    print("Running preprocessing pipeline...")
    result = preprocess_handwriting(img)
    print(f"  Result shape:   {result.shape}, dtype: {result.dtype}")

    if args.output:
        save_image(result, args.output)
        print(f"Saved to: {args.output}")
    else:
        cv2.imshow("Preprocessed", result)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
