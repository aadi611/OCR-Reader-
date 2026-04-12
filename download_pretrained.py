"""
Download PP-OCRv4 pretrained models from PaddleOCR's BOS (Baidu Object Storage).

Downloads:
  - en_PP-OCRv4_det_train.tar   (text detection model)
  - en_PP-OCRv4_rec_train.tar   (text recognition model)

Both archives are extracted to ./pretrain_models/ and the .tar files are
removed after successful extraction.

Usage:
    python download_pretrained.py
    python download_pretrained.py --output-dir /path/to/pretrain_models
    python download_pretrained.py --det-only
    python download_pretrained.py --rec-only
    python download_pretrained.py --keep-tar   # keep .tar after extraction
"""

import argparse
import hashlib
import os
import sys
import tarfile
import time
import urllib.request
from typing import Optional


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Official PaddleOCR BOS URLs (PP-OCRv4 English models)
_MODELS = {
    "det": {
        "url": (
            "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/"
            "ch_PP-OCRv4_det_train.tar"
        ),
        "filename": "ch_PP-OCRv4_det_train.tar",
        "extract_dir": "ch_PP-OCRv4_det_train",
        "description": "PP-OCRv4 Detection (DB ResNet-50, supports English + multilingual)",
    },
    "rec": {
        "url": (
            "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/"
            "en_PP-OCRv4_rec_train.tar"
        ),
        "filename": "en_PP-OCRv4_rec_train.tar",
        "extract_dir": "en_PP-OCRv4_rec_train",
        "description": "PP-OCRv4 English Recognition (SVTR_LCNet)",
    },
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PP-OCRv4 pretrained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="./pretrain_models",
        help="Directory where models will be saved and extracted.",
    )
    parser.add_argument(
        "--det-only",
        action="store_true",
        default=False,
        help="Download only the detection model.",
    )
    parser.add_argument(
        "--rec-only",
        action="store_true",
        default=False,
        help="Download only the recognition model.",
    )
    parser.add_argument(
        "--keep-tar",
        action="store_true",
        default=False,
        help="Keep the .tar archive after extraction (deleted by default).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download even if the target directory already exists.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Progress-reporting download
# ---------------------------------------------------------------------------

class _DownloadProgressBar:
    """Simple console progress bar for urllib.request.urlretrieve."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self._last_pct = -1
        self._start = time.time()

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            downloaded = block_num * block_size
            mb = downloaded / (1024 ** 2)
            sys.stdout.write(f"\r  Downloading {self.filename} ... {mb:.1f} MB")
            sys.stdout.flush()
            return

        downloaded = min(block_num * block_size, total_size)
        pct = int(downloaded * 100 / total_size)

        if pct == self._last_pct:
            return
        self._last_pct = pct

        bar_width = 40
        filled = int(bar_width * pct / 100)
        bar = "=" * filled + "-" * (bar_width - filled)
        elapsed = time.time() - self._start
        mb_done = downloaded / (1024 ** 2)
        mb_total = total_size / (1024 ** 2)

        speed_str = ""
        if elapsed > 0:
            speed = mb_done / elapsed
            speed_str = f"  {speed:.1f} MB/s"

        sys.stdout.write(
            f"\r  [{bar}] {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MB{speed_str}  "
        )
        sys.stdout.flush()

        if pct == 100:
            print()  # newline after completion


# ---------------------------------------------------------------------------
# Core download + extract logic
# ---------------------------------------------------------------------------

def download_file(url: str, dest_path: str, filename: str) -> None:
    """
    Download a file with a progress bar.

    Args:
        url:       Remote URL.
        dest_path: Local filesystem path to write to.
        filename:  Display name for the progress bar.
    """
    progress = _DownloadProgressBar(filename)
    print(f"  URL : {url}")
    print(f"  Dest: {dest_path}")
    urllib.request.urlretrieve(url, dest_path, reporthook=progress)


def extract_tar(tar_path: str, output_dir: str) -> None:
    """
    Extract a .tar (or .tar.gz / .tar.bz2) archive.

    Args:
        tar_path:   Path to the archive file.
        output_dir: Directory where contents are extracted.
    """
    print(f"  Extracting {os.path.basename(tar_path)} ...")
    t0 = time.time()
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=output_dir)
    elapsed = time.time() - t0
    print(f"  Extracted in {elapsed:.1f}s -> {output_dir}")


def _list_extracted_files(directory: str, max_files: int = 8) -> None:
    """Print a summary of files found in an extracted directory."""
    if not os.path.isdir(directory):
        print(f"    (directory not found: {directory})")
        return

    all_files = []
    for root, _, files in os.walk(directory):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, directory)
            size_mb = os.path.getsize(fpath) / (1024 ** 2)
            all_files.append((rel, size_mb))

    all_files.sort(key=lambda x: x[1], reverse=True)
    for rel, size_mb in all_files[:max_files]:
        print(f"    {rel:<55} {size_mb:>8.2f} MB")
    if len(all_files) > max_files:
        print(f"    ... and {len(all_files) - max_files} more files")


def download_model(
    key: str,
    output_dir: str,
    keep_tar: bool = False,
    force: bool = False,
) -> bool:
    """
    Download and extract a single model.

    Args:
        key:        Model key in _MODELS ("det" or "rec").
        output_dir: Base directory for pretrained models.
        keep_tar:   Whether to keep the .tar file after extraction.
        force:      Re-download even if already present.

    Returns:
        True on success, False on failure.
    """
    info = _MODELS[key]
    url = info["url"]
    filename = info["filename"]
    extract_subdir = info["extract_dir"]
    description = info["description"]

    os.makedirs(output_dir, exist_ok=True)

    tar_path = os.path.join(output_dir, filename)
    extract_path = os.path.join(output_dir, extract_subdir)

    print()
    print(f"  Model      : {description}")

    # Check if already extracted
    if os.path.isdir(extract_path) and not force:
        best_path = os.path.join(extract_path, "best_accuracy.pdparams")
        if os.path.isfile(best_path):
            print(f"  Status     : Already present at {extract_path} (skip).")
            print(f"               Use --force to re-download.")
            return True
        else:
            print(f"  Status     : Directory exists but may be incomplete; re-downloading.")

    # Download
    try:
        download_file(url, tar_path, filename)
    except Exception as exc:
        print(f"\n  [ERROR] Download failed: {exc}", file=sys.stderr)
        if os.path.isfile(tar_path):
            os.remove(tar_path)
        return False

    tar_size_mb = os.path.getsize(tar_path) / (1024 ** 2)
    print(f"  Downloaded : {tar_size_mb:.1f} MB")

    # Extract
    try:
        extract_tar(tar_path, output_dir)
    except Exception as exc:
        print(f"  [ERROR] Extraction failed: {exc}", file=sys.stderr)
        return False

    # Show extracted contents
    print(f"\n  Extracted files in {extract_path}:")
    _list_extracted_files(extract_path)

    # Cleanup
    if not keep_tar:
        os.remove(tar_path)
        print(f"\n  Removed archive: {filename}")
    else:
        print(f"\n  Archive kept  : {tar_path}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("  PP-OCRv4 Pretrained Model Downloader")
    print("=" * 72)
    print(f"  Output dir : {os.path.abspath(args.output_dir)}")
    print(f"  Keep tar   : {args.keep_tar}")
    print(f"  Force      : {args.force}")
    print("=" * 72)

    to_download: list = []
    if args.det_only:
        to_download = ["det"]
    elif args.rec_only:
        to_download = ["rec"]
    else:
        to_download = ["det", "rec"]

    results: dict = {}
    for key in to_download:
        print(f"\n--- Downloading {key.upper()} model ---")
        success = download_model(
            key=key,
            output_dir=args.output_dir,
            keep_tar=args.keep_tar,
            force=args.force,
        )
        results[key] = success

    print()
    print("=" * 72)
    print("  Summary")
    print("=" * 72)
    all_ok = True
    for key, success in results.items():
        status = "OK" if success else "FAILED"
        model_dir = os.path.join(args.output_dir, _MODELS[key]["extract_dir"])
        print(f"  {key.upper():<4} : {status:<8}  {model_dir}")
        if not success:
            all_ok = False
    print("=" * 72)

    if all_ok:
        print("\nAll models downloaded and ready.")
        print(
            "\nNext step: update config files to point to pretrained model paths, "
            "then run:\n    python train.py --stage both"
        )
    else:
        print("\nSome downloads failed. Check errors above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
