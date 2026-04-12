"""
Export trained PaddleOCR models to inference format and/or ONNX.

Usage:
    # Export a detection model to both Paddle inference and ONNX formats
    python export.py --model-dir ./output/det_handwriting/best_accuracy \\
                     --config ./configs/det/handwriting_det.yml \\
                     --output-dir ./inference/det \\
                     --format both

    # Export recognition model to Paddle inference only
    python export.py --model-dir ./output/rec_handwriting_svtr/best_accuracy \\
                     --config ./configs/rec/handwriting_rec_svtr.yml \\
                     --output-dir ./inference/rec \\
                     --format paddle

Dependencies:
    paddle2onnx must be installed for ONNX export:
        pip install paddle2onnx onnxruntime
"""

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PaddleOCR models to inference or ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing the trained model (best_accuracy or a checkpoint).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training YAML config corresponding to this model.",
    )
    parser.add_argument(
        "--output-dir",
        default="./inference",
        help="Destination directory for exported model files.",
    )
    parser.add_argument(
        "--format",
        choices=["paddle", "onnx", "both"],
        default="both",
        help="Export format: Paddle inference model, ONNX, or both.",
    )
    parser.add_argument(
        "--paddle-export-script",
        default=None,
        help="Explicit path to PaddleOCR's tools/export_model.py (auto-detected if not set).",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="ONNX opset version for paddle2onnx conversion.",
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use GPU during export (not usually needed).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def find_export_script(explicit_path: Optional[str] = None) -> str:
    """Locate PaddleOCR's tools/export_model.py."""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path

    try:
        import paddleocr as _poc

        package_dir = os.path.dirname(_poc.__file__)
        for candidate in [
            os.path.join(package_dir, "tools", "export_model.py"),
            os.path.normpath(os.path.join(package_dir, "..", "tools", "export_model.py")),
        ]:
            if os.path.isfile(candidate):
                return candidate
    except ImportError:
        pass

    local = os.path.join(os.getcwd(), "PaddleOCR", "tools", "export_model.py")
    if os.path.isfile(local):
        return local

    raise FileNotFoundError(
        "Cannot locate PaddleOCR tools/export_model.py. "
        "Pass --paddle-export-script explicitly."
    )


def _format_size(path: str) -> str:
    """Return human-readable size of a file."""
    if not os.path.isfile(path):
        return "N/A"
    size = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _print_output_files(directory: str) -> None:
    """Print all files in directory with sizes."""
    if not os.path.isdir(directory):
        print(f"  (directory not found: {directory})")
        return
    files = sorted(os.listdir(directory))
    if not files:
        print("  (no files)")
        return
    for fname in files:
        fpath = os.path.join(directory, fname)
        size = _format_size(fpath)
        print(f"  {fname:<45} {size:>10}")


# ---------------------------------------------------------------------------
# Paddle inference export
# ---------------------------------------------------------------------------

def export_paddle(
    export_script: str,
    model_dir: str,
    config_path: str,
    output_dir: str,
    use_gpu: bool = False,
) -> int:
    """
    Export to Paddle static inference model using tools/export_model.py.

    Generates:
        <output_dir>/inference.pdmodel
        <output_dir>/inference.pdiparams
        <output_dir>/inference.pdiparams.info

    Returns:
        Subprocess exit code (0 = success).
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, export_script,
        "-c", config_path,
        "-o",
        f"Global.pretrained_model={model_dir}",
        f"Global.save_inference_dir={output_dir}",
        f"Global.use_gpu={'true' if use_gpu else 'false'}",
    ]

    print(f"\n  Command: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  Paddle export: {status} ({elapsed:.1f}s)")

    if result.returncode == 0:
        print(f"\n  Output files in {output_dir}:")
        _print_output_files(output_dir)

    return result.returncode


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    paddle_inference_dir: str,
    onnx_output_dir: str,
    model_filename: str = "inference.pdmodel",
    params_filename: str = "inference.pdiparams",
    opset_version: int = 11,
) -> int:
    """
    Convert a Paddle inference model to ONNX using paddle2onnx.

    Args:
        paddle_inference_dir: Directory containing .pdmodel and .pdiparams files.
        onnx_output_dir:      Directory where the .onnx file will be written.
        model_filename:       Name of the pdmodel file.
        params_filename:      Name of the pdiparams file.
        opset_version:        ONNX opset version.

    Returns:
        Exit code (0 = success).
    """
    try:
        import paddle2onnx  # noqa: F401 — just check it's installed
    except ImportError:
        print(
            "  [ERROR] paddle2onnx is not installed. "
            "Run: pip install paddle2onnx",
            file=sys.stderr,
        )
        return 1

    os.makedirs(onnx_output_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_output_dir, "model.onnx")

    model_file = os.path.join(paddle_inference_dir, model_filename)
    params_file = os.path.join(paddle_inference_dir, params_filename)

    if not os.path.isfile(model_file):
        print(f"  [ERROR] pdmodel not found: {model_file}", file=sys.stderr)
        return 1
    if not os.path.isfile(params_file):
        print(f"  [ERROR] pdiparams not found: {params_file}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, "-m", "paddle2onnx",
        "--model_dir", paddle_inference_dir,
        "--model_filename", model_filename,
        "--params_filename", params_filename,
        "--save_file", onnx_path,
        "--opset_version", str(opset_version),
        "--enable_onnx_checker", "True",
    ]

    print(f"\n  Command: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  ONNX export: {status} ({elapsed:.1f}s)")

    if result.returncode == 0 and os.path.isfile(onnx_path):
        size = _format_size(onnx_path)
        print(f"\n  ONNX model: {onnx_path}  ({size})")

        # Optionally validate with onnxruntime
        _validate_onnx(onnx_path)

    return result.returncode


def _validate_onnx(onnx_path: str) -> None:
    """Run a quick ONNX model validation if onnxruntime is available."""
    try:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # suppress info logs
        session = ort.InferenceSession(onnx_path, sess_options=sess_options)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        print(f"  ONNX validation: OK")
        print(f"    Inputs : {[f'{i.name} {i.shape}' for i in inputs]}")
        print(f"    Outputs: {[f'{o.name} {o.shape}' for o in outputs]}")
    except ImportError:
        print("  [INFO] onnxruntime not installed; skipping ONNX validation.")
    except Exception as exc:
        print(f"  [WARN] ONNX validation failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("  PaddleOCR Model Export")
    print("=" * 72)
    print(f"  Model dir  : {args.model_dir}")
    print(f"  Config     : {args.config}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Format     : {args.format}")
    print(f"  Opset      : {args.opset_version}")
    print("=" * 72)

    # Validate inputs
    if not os.path.isfile(args.config):
        print(f"[ERROR] Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Locate export script
    try:
        export_script = find_export_script(args.paddle_export_script)
        print(f"Export script: {export_script}\n")
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    paddle_output_dir = os.path.join(args.output_dir, "paddle")
    onnx_output_dir = os.path.join(args.output_dir, "onnx")

    overall_exit = 0

    # --- Paddle inference export ---
    if args.format in ("paddle", "both"):
        print("\n[Step 1/2] Exporting to Paddle inference format...")
        rc = export_paddle(
            export_script=export_script,
            model_dir=args.model_dir,
            config_path=args.config,
            output_dir=paddle_output_dir,
            use_gpu=args.use_gpu,
        )
        if rc != 0:
            overall_exit = rc

    # --- ONNX export ---
    if args.format in ("onnx", "both"):
        step = "2/2" if args.format == "both" else "1/1"
        print(f"\n[Step {step}] Exporting to ONNX format...")

        if args.format == "both" and overall_exit != 0:
            print("  [WARN] Paddle export failed; ONNX conversion may also fail.")

        # For "onnx only", we still need a Paddle inference model first
        if args.format == "onnx":
            print("  Building Paddle inference model first (required for ONNX)...")
            rc = export_paddle(
                export_script=export_script,
                model_dir=args.model_dir,
                config_path=args.config,
                output_dir=paddle_output_dir,
                use_gpu=args.use_gpu,
            )
            if rc != 0:
                print(
                    "  [ERROR] Paddle export step failed; cannot produce ONNX.",
                    file=sys.stderr,
                )
                sys.exit(rc)

        rc = export_onnx(
            paddle_inference_dir=paddle_output_dir,
            onnx_output_dir=onnx_output_dir,
            opset_version=args.opset_version,
        )
        if rc != 0:
            overall_exit = rc

    print("\n" + "=" * 72)
    if overall_exit == 0:
        print("  Export completed successfully.")
    else:
        print(f"  Export finished with errors (exit code {overall_exit}).")
    print("=" * 72)

    if overall_exit != 0:
        sys.exit(overall_exit)


if __name__ == "__main__":
    main()
