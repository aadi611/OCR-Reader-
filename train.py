"""
Training wrapper script for PaddleOCR handwriting models.

Runs detection and/or recognition training by calling PaddleOCR's
tools/train.py with the appropriate configuration and GPU arguments.

Usage:
    # Train both detection and recognition (sequential)
    python train.py --stage both

    # Train detection only on GPU 0
    python train.py --stage det --gpus 0

    # Train recognition on GPUs 0 and 1 with a custom config
    python train.py --stage rec --gpus 0,1 --config-rec configs/rec/my_rec.yml

    # Resume from a checkpoint
    python train.py --stage rec --resume
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
        description="Train PaddleOCR detection and/or recognition models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["det", "rec", "both"],
        default="both",
        help="Which stage to train: detection, recognition, or both (sequential).",
    )
    parser.add_argument(
        "--config-det",
        default="./configs/det/handwriting_det.yml",
        help="Path to the detection config YAML.",
    )
    parser.add_argument(
        "--config-rec",
        default="./configs/rec/handwriting_rec_svtr.yml",
        help="Path to the recognition config YAML.",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU IDs to use (e.g. '0' or '0,1,2,3').",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the latest checkpoint in save_model_dir.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Run evaluation only (no training).",
    )
    parser.add_argument(
        "--paddle-train-script",
        default=None,
        help="Explicit path to PaddleOCR's tools/train.py. "
             "Auto-detected from paddleocr package if not set.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Locate PaddleOCR train script
# ---------------------------------------------------------------------------

def find_paddle_train_script(explicit_path: Optional[str] = None) -> str:
    """
    Find the path to PaddleOCR's tools/train.py.

    Checks (in order):
      1. The explicit path argument.
      2. The paddleocr package installation directory.
      3. A local PaddleOCR checkout in the current working directory.

    Raises:
        FileNotFoundError: if the script cannot be located.
    """
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path

    # Try the installed paddleocr package location
    try:
        import paddleocr as _poc

        package_dir = os.path.dirname(_poc.__file__)
        candidate = os.path.join(package_dir, "tools", "train.py")
        if os.path.isfile(candidate):
            return candidate

        # Some installations place it one level up
        candidate2 = os.path.join(package_dir, "..", "tools", "train.py")
        candidate2 = os.path.normpath(candidate2)
        if os.path.isfile(candidate2):
            return candidate2
    except ImportError:
        pass

    # Try a local PaddleOCR source checkout
    local_candidate = os.path.join(os.getcwd(), "PaddleOCR", "tools", "train.py")
    if os.path.isfile(local_candidate):
        return local_candidate

    raise FileNotFoundError(
        "Cannot locate PaddleOCR tools/train.py. "
        "Please install paddleocr or pass --paddle-train-script."
    )


# ---------------------------------------------------------------------------
# GPU environment setup
# ---------------------------------------------------------------------------

def build_gpu_env(gpus: str) -> dict:
    """
    Build environment variables for multi-GPU training.

    Returns a copy of os.environ with CUDA_VISIBLE_DEVICES set.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    return env


def get_gpu_count(gpus: str) -> int:
    """Count the number of GPUs specified."""
    return len([g.strip() for g in gpus.split(",") if g.strip()])


# ---------------------------------------------------------------------------
# Build training command
# ---------------------------------------------------------------------------

def build_train_command(
    train_script: str,
    config_path: str,
    gpus: str,
    resume: bool = False,
    eval_only: bool = False,
) -> List[str]:
    """
    Construct the command list for subprocess to run PaddleOCR training.

    Args:
        train_script: Path to tools/train.py.
        config_path:  Path to the YAML config file.
        gpus:         Comma-separated GPU IDs string.
        resume:       If True, pass -c checkpoint flag to resume training.
        eval_only:    If True, pass -o eval flag.

    Returns:
        List of command tokens.
    """
    n_gpus = get_gpu_count(gpus)

    if n_gpus > 1:
        # Multi-GPU: use paddle.distributed.launch
        cmd = [
            sys.executable, "-m", "paddle.distributed.launch",
            f"--gpus={gpus}",
            train_script,
            "-c", config_path,
        ]
    else:
        cmd = [sys.executable, train_script, "-c", config_path]

    if eval_only:
        cmd += ["-o", "Global.infer_mode=true"]

    if resume:
        # Instruct train.py to resume from the latest checkpoint
        cmd += ["-o", "Global.checkpoints=latest"]

    return cmd


# ---------------------------------------------------------------------------
# Run a single training stage
# ---------------------------------------------------------------------------

def run_stage(
    stage_name: str,
    train_script: str,
    config_path: str,
    gpus: str,
    resume: bool,
    eval_only: bool,
) -> int:
    """
    Execute one training stage and return the subprocess exit code.
    """
    if not os.path.isfile(config_path):
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 1

    cmd = build_train_command(train_script, config_path, gpus, resume, eval_only)
    env = build_gpu_env(gpus)
    n_gpus = get_gpu_count(gpus)

    mode = "Evaluation" if eval_only else "Training"
    print()
    print("=" * 72)
    print(f"  {mode} Stage: {stage_name.upper()}")
    print(f"  Config  : {config_path}")
    print(f"  GPUs    : {gpus} ({n_gpus} device(s))")
    print(f"  Resume  : {resume}")
    print(f"  Command : {' '.join(cmd)}")
    print("=" * 72)

    t0 = time.time()
    try:
        result = subprocess.run(cmd, env=env, check=False)
        exit_code = result.returncode
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {stage_name} training interrupted by user.")
        exit_code = 130
    except Exception as exc:
        print(f"[ERROR] Failed to launch {stage_name} training: {exc}", file=sys.stderr)
        exit_code = 1

    elapsed = time.time() - t0
    hours, rem = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(rem, 60)

    status = "SUCCESS" if exit_code == 0 else f"FAILED (exit {exit_code})"
    print()
    print(f"  {stage_name.upper()} training finished: {status}")
    print(f"  Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print("=" * 72)

    return exit_code


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Locate train.py
    try:
        train_script = find_paddle_train_script(args.paddle_train_script)
        print(f"PaddleOCR train script: {train_script}")
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    stages_to_run: List[str] = []
    if args.stage in ("det", "both"):
        stages_to_run.append("det")
    if args.stage in ("rec", "both"):
        stages_to_run.append("rec")

    config_map = {
        "det": args.config_det,
        "rec": args.config_rec,
    }

    overall_exit = 0
    for stage in stages_to_run:
        exit_code = run_stage(
            stage_name=stage,
            train_script=train_script,
            config_path=config_map[stage],
            gpus=args.gpus,
            resume=args.resume,
            eval_only=args.eval_only,
        )
        if exit_code != 0:
            overall_exit = exit_code
            print(
                f"[WARN] Stage '{stage}' failed with exit code {exit_code}. "
                "Continuing to next stage..." if len(stages_to_run) > 1 else "",
                file=sys.stderr,
            )

    if overall_exit == 0:
        print("\nAll requested training stages completed successfully.")
    else:
        print(f"\nOne or more training stages failed (exit code {overall_exit}).", file=sys.stderr)
        sys.exit(overall_exit)


if __name__ == "__main__":
    main()
