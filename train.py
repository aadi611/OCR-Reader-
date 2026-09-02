"""
Training wrapper for PaddleOCR handwriting models.

Drives PaddleOCR's tools/train.py (or tools/eval.py) with the right config,
GPU topology and override flags, streams the child's output to both the
terminal and a log file, and shuts the child down cleanly on Ctrl-C.

Usage:
    python train.py --stage both                       # det then rec
    python train.py --stage det --gpus 0
    python train.py --stage rec --gpus 0,1 --amp       # mixed precision
    python train.py --stage rec --resume               # from latest checkpoint
    python train.py --stage rec --eval-only            # runs tools/eval.py
    python train.py --stage both --dry-run             # print commands, do nothing
    python train.py --stage rec -o Global.epoch_num=50 Optimizer.lr.learning_rate=0.0005
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

IS_POSIX = os.name == "posix"


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PaddleOCR detection and/or recognition models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=["det", "rec", "both"], default="both",
                        help="Detection, recognition, or both (sequential).")
    parser.add_argument("--config-det", default="./configs/det/handwriting_det.yml",
                        help="Detection config YAML.")
    parser.add_argument("--config-rec", default="./configs/rec/handwriting_rec_svtr.yml",
                        help="Recognition config YAML.")
    parser.add_argument("--gpus", default="0",
                        help="Comma-separated GPU IDs, or 'cpu' to train on CPU.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <save_model_dir>/latest.")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit checkpoint prefix; overrides --resume's guess.")
    parser.add_argument("--pretrained", default=None,
                        help="Pretrained weights prefix (Global.pretrained_model).")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run tools/eval.py instead of training.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision (usually 1.5-2x faster on modern GPUs).")
    parser.add_argument("--keep-going", action="store_true",
                        help="Run later stages even if an earlier one fails.")
    parser.add_argument("--log-dir", default="./logs",
                        help="Directory for per-stage log files. Empty string disables.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands that would run, then exit.")
    parser.add_argument("--paddle-train-script", default=None,
                        help="Explicit path to PaddleOCR's tools/train.py.")
    parser.add_argument("-o", "--opt", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Extra config overrides passed straight through to PaddleOCR.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Locating PaddleOCR's tools/
# ---------------------------------------------------------------------------
def find_tools_dir(explicit_train_script: str | None = None) -> Path:
    """
    Return the directory holding train.py / eval.py.

    Uses importlib's finder rather than importing paddleocr — importing the
    package costs several seconds and pulls in paddle just to read a path.
    """
    if explicit_train_script:
        p = Path(explicit_train_script).expanduser()
        if p.is_file():
            return p.parent
        raise FileNotFoundError(f"--paddle-train-script not found: {p}")

    candidates: list[Path] = []

    env_home = os.environ.get("PADDLEOCR_HOME")
    if env_home:
        candidates.append(Path(env_home) / "tools")

    spec = importlib.util.find_spec("paddleocr")
    origin = getattr(spec, "origin", None) if spec else None
    if origin:
        pkg = Path(origin).parent
        candidates += [pkg / "tools", pkg.parent / "tools"]

    cwd = Path.cwd()
    candidates += [cwd / "PaddleOCR" / "tools", cwd / "tools"]

    for c in candidates:
        if (c / "train.py").is_file():
            return c.resolve()

    raise FileNotFoundError(
        "Cannot locate PaddleOCR tools/train.py. Install paddleocr, set "
        "PADDLEOCR_HOME, or pass --paddle-train-script."
    )


# ---------------------------------------------------------------------------
# Config inspection
# ---------------------------------------------------------------------------
_SAVE_DIR_RE = re.compile(r"^\s*save_model_dir\s*:\s*[\"']?([^\"'#\n]+)", re.M)


def read_save_model_dir(config_path: Path) -> str | None:
    """Pull Global.save_model_dir out of the YAML (PyYAML optional)."""
    try:
        import yaml  # noqa: PLC0415

        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        value = (cfg.get("Global") or {}).get("save_model_dir")
        if value:
            return str(value).strip()
    except ImportError:
        pass
    except Exception as exc:
        print(f"[WARN] could not parse {config_path}: {exc}", file=sys.stderr)

    try:
        match = _SAVE_DIR_RE.search(config_path.read_text(encoding="utf-8"))
        return match.group(1).strip() if match else None
    except OSError:
        return None


def resolve_checkpoint(config_path: Path, explicit: str | None) -> str | None:
    """
    Work out the checkpoint prefix to resume from.

    PaddleOCR wants a path prefix such as ./output/rec/latest — the literal
    string "latest" is not a valid value and silently starts from scratch.
    """
    if explicit:
        return explicit
    save_dir = read_save_model_dir(config_path)
    if not save_dir:
        print("[WARN] --resume: no Global.save_model_dir in config; "
              "pass --checkpoint explicitly.", file=sys.stderr)
        return None
    prefix = Path(save_dir) / "latest"
    if not prefix.with_suffix(".pdparams").is_file():
        print(f"[WARN] --resume: no checkpoint at {prefix}.pdparams — "
              "training will start from scratch.", file=sys.stderr)
        return None
    return str(prefix)


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------
def gpu_ids(gpus: str) -> list[str]:
    if gpus.strip().lower() in ("", "cpu", "none", "-1"):
        return []
    return [g.strip() for g in gpus.split(",") if g.strip()]


def build_command(
    tools_dir: Path, config_path: Path, args: argparse.Namespace
) -> tuple[list[str], dict[str, str]]:
    """Build the argv and environment for one stage."""
    script = tools_dir / ("eval.py" if args.eval_only else "train.py")
    if not script.is_file():
        raise FileNotFoundError(f"{script} not found next to train.py")

    ids = gpu_ids(args.gpus)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")       # stream child logs live
    env.setdefault("FLAGS_allocator_strategy", "auto_growth")

    if not ids:
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [sys.executable, str(script)]
    elif len(ids) == 1:
        env["CUDA_VISIBLE_DEVICES"] = ids[0]
        cmd = [sys.executable, str(script)]
    else:
        # Let launch own the device selection. Setting CUDA_VISIBLE_DEVICES as
        # well makes --gpus indices relative to that mask and re-maps twice.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        cmd = [sys.executable, "-m", "paddle.distributed.launch",
               f"--gpus={','.join(ids)}", str(script)]

    cmd += ["-c", str(config_path)]

    # PaddleOCR's -o takes nargs='+' with a plain store action, so a second -o
    # REPLACES the first. Every override has to go in one list.
    overrides: list[str] = []
    if not ids:
        overrides.append("Global.use_gpu=False")
    if args.amp and ids:
        overrides += ["Global.use_amp=True", "Global.scale_loss=1024.0",
                      "Global.use_dynamic_loss_scaling=True"]
    if args.pretrained:
        overrides.append(f"Global.pretrained_model={args.pretrained}")
    ckpt = (resolve_checkpoint(config_path, args.checkpoint)
            if (args.resume or args.checkpoint or args.eval_only) else None)
    if ckpt:
        overrides.append(f"Global.checkpoints={ckpt}")
    overrides += list(args.opt)
    if overrides:
        cmd += ["-o", *overrides]

    return cmd, env


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    name: str
    exit_code: int
    elapsed: float
    log_path: Path | None = None
    skipped: bool = False


_child: subprocess.Popen | None = None
_interrupted = False


def _forward_signal(signum, _frame) -> None:
    """Pass Ctrl-C / SIGTERM to the training process instead of orphaning it."""
    global _interrupted
    _interrupted = True
    proc = _child
    if proc and proc.poll() is None:
        try:
            if IS_POSIX:
                os.killpg(os.getpgid(proc.pid), signum)
            else:
                proc.terminate()
        except (ProcessLookupError, PermissionError):
            pass


def run_command(cmd: list[str], env: dict[str, str], log_path: Path | None) -> int:
    """Run the child, teeing its output to the terminal and (optionally) a file."""
    global _child

    stdout = subprocess.PIPE if log_path else None
    log_file = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "wb", buffering=0)

    popen_kwargs: dict = {"env": env, "stdout": stdout}
    if log_path:
        popen_kwargs["stderr"] = subprocess.STDOUT
    if IS_POSIX:
        popen_kwargs["start_new_session"] = True   # own group, so we can signal it
    else:
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    previous = {
        sig: signal.signal(sig, _forward_signal)
        for sig in (signal.SIGINT, signal.SIGTERM)
        if hasattr(signal, sig.name)
    }
    try:
        _child = subprocess.Popen(cmd, **popen_kwargs)
        if log_file is not None and _child.stdout is not None:
            out = sys.stdout.buffer
            # Chunked, not line-based: keeps \r progress bars rendering properly.
            while chunk := _child.stdout.read(4096):
                out.write(chunk)
                out.flush()
                log_file.write(chunk)
        code = _child.wait()
        if code != 0 and _interrupted:
            code = 130
        return code
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if log_file is not None:
            log_file.close()
        _child = None


def run_stage(name: str, tools_dir: Path, config_path: Path,
              args: argparse.Namespace) -> StageResult:
    cmd, env = build_command(tools_dir, config_path, args)
    ids = gpu_ids(args.gpus)
    mode = "Evaluation" if args.eval_only else "Training"
    log_path = (Path(args.log_dir) / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
                if args.log_dir else None)

    print("\n" + "=" * 72)
    print(f"  {mode} stage : {name.upper()}")
    print(f"  Config       : {config_path}")
    print(f"  Devices      : {', '.join(ids) if ids else 'CPU'}"
          f"{f' ({len(ids)} GPUs, distributed)' if len(ids) > 1 else ''}")
    print(f"  AMP          : {bool(args.amp and ids)}")
    if log_path:
        print(f"  Log          : {log_path}")
    print(f"  Command      : {' '.join(cmd)}")
    print("=" * 72, flush=True)

    if args.dry_run:
        return StageResult(name, 0, 0.0, log_path, skipped=True)

    t0 = time.monotonic()
    try:
        code = run_command(cmd, env, log_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] cannot launch {name}: {exc}", file=sys.stderr)
        code = 127
    elapsed = time.monotonic() - t0

    status = "SUCCESS" if code == 0 else ("INTERRUPTED" if code == 130 else f"FAILED ({code})")
    print(f"\n  {name.upper()} {mode.lower()} finished: {status}")
    print(f"  Elapsed: {format_hms(elapsed)}")
    print("=" * 72, flush=True)
    return StageResult(name, code, elapsed, log_path)


def format_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def check_gpus(ids: Sequence[str]) -> None:
    """Warn about missing GPUs now rather than 30 seconds into a doomed launch."""
    if not ids or not shutil.which("nvidia-smi"):
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return
    available = {line.strip() for line in out.splitlines() if line.strip()}
    missing = [g for g in ids if g not in available]
    if missing:
        print(f"[WARN] GPU(s) {','.join(missing)} not visible to nvidia-smi "
              f"(present: {','.join(sorted(available)) or 'none'}).", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        tools_dir = find_tools_dir(args.paddle_train_script)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"PaddleOCR tools: {tools_dir}")

    stages = ["det", "rec"] if args.stage == "both" else [args.stage]
    configs = {"det": Path(args.config_det), "rec": Path(args.config_rec)}

    # Validate every config before the first stage — a missing rec config
    # should not surface six hours into det training.
    missing = [str(configs[s]) for s in stages if not configs[s].is_file()]
    if missing:
        print(f"[ERROR] config file(s) not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    check_gpus(gpu_ids(args.gpus))

    results: list[StageResult] = []
    for stage in stages:
        result = run_stage(stage, tools_dir, configs[stage], args)
        results.append(result)
        if result.exit_code == 130:
            print("[INTERRUPTED] stopping; remaining stages skipped.", file=sys.stderr)
            break
        if result.exit_code != 0 and not args.keep_going:
            print(f"[ERROR] stage '{stage}' failed (exit {result.exit_code}); "
                  "remaining stages skipped. Use --keep-going to override.",
                  file=sys.stderr)
            break

    if len(results) > 1 or any(r.exit_code for r in results):
        print("\n" + "-" * 72)
        print(f"  {'Stage':<8} {'Status':<14} {'Elapsed':>10}  Log")
        print("-" * 72)
        for r in results:
            status = ("DRY-RUN" if r.skipped else
                      "SUCCESS" if r.exit_code == 0 else
                      "INTERRUPTED" if r.exit_code == 130 else f"FAILED ({r.exit_code})")
            print(f"  {r.name.upper():<8} {status:<14} {format_hms(r.elapsed):>10}  "
                  f"{r.log_path or '-'}")
        print("-" * 72)

    worst = next((r.exit_code for r in results if r.exit_code), 0)
    if worst == 0:
        print("\nAll requested stages completed successfully.")
    else:
        print(f"\nOne or more stages did not succeed (exit {worst}).", file=sys.stderr)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
