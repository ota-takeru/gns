#!/usr/bin/env python3
"""Utility script to run the full GNS pipeline on Kaggle.

This orchestrates dependency installation, dataset generation, training, and
rollout inference using the lighter Kaggle-oriented configuration files that
live in the repository. The script prefers CPU execution to avoid driver
compatibility issues inside Kaggle containers, but you can override that by
setting CUDA_VISIBLE_DEVICES before invoking it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = REPO_ROOT / "datasets" / "config" / "fluid_kaggle.yaml"
TRAIN_CONFIG = REPO_ROOT / "config_kaggle.yaml"
ROLLOUT_CONFIG = REPO_ROOT / "config_kaggle_rollout.yaml"
DEPS_SENTINEL = REPO_ROOT / "kaggle" / ".deps_installed"


def _run_command(cmd: Sequence[str], env: dict[str, str] | None = None) -> None:
    """Run a subprocess while echoing the command for notebook logs."""
    display_cmd = " ".join(cmd)
    print(f"[pipeline] $ {display_cmd}")
    subprocess.check_call(list(cmd), cwd=REPO_ROOT, env=env)


def _pip_install(args: Iterable[str]) -> None:
    base = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
    _run_command([*base, *args])


def install_dependencies(force: bool = False) -> None:
    """Install the minimal set of packages required for Kaggle execution."""
    if DEPS_SENTINEL.exists() and not force:
        print("[pipeline] Dependencies already installed, skipping.")
        return

    # print("[pipeline] Installing PyTorch (CPU wheels).")
    # _pip_install(
    #     [
    #         "--index-url",
    #         "https://download.pytorch.org/whl/cpu",
    #         "torch==2.5.1+cpu",
    #         "torchvision==0.20.1+cpu",
    #         "torchaudio==2.5.1",
    #     ]
    # )

    # print("[pipeline] Installing PyG extension wheels.")
    # _pip_install(
    #     [
    #         "pyg_lib",
    #         "torch_scatter",
    #         "torch_sparse",
    #         "torch_cluster",
    #         "torch_spline_conv",
    #         "-f",
    #         "https://data.pyg.org/whl/torch-2.5.0+cpu.html",
    #     ]
    # )

    print("[pipeline] Installing project Python dependencies.")
    _pip_install(
        [
            "torch-geometric==2.5.3",
            "pysph==1.0b2",
            "pymunk==7.1.0",
            "pyyaml==6.0.2",
            "h5py==3.11.0",
            "matplotlib==3.8.4",
            "tensorboard==2.16.2",
            "tqdm==4.66.4",
        ]
    )

    DEPS_SENTINEL.write_text("installed\n", encoding="utf-8")
    print("[pipeline] Dependency installation complete.")


def _ensure_config(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        msg = f"{label} not found: {resolved}"
        raise FileNotFoundError(msg)
    return resolved


def generate_dataset(config_path: Path) -> None:
    """Run the PySPH-based fluid dataset generator."""
    cfg = _ensure_config(config_path, "Dataset config")
    _run_command([sys.executable, "datasets/scripts/gen_fluid.py", "--config", str(cfg)])


def train_model(config_path: Path) -> None:
    """Train the learned simulator using the Kaggle-friendly config."""
    cfg = _ensure_config(config_path, "Training config")
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    _run_command([sys.executable, "src/train.py", "--config", str(cfg)], env=env)


def run_rollout(config_path: Path) -> None:
    """Execute rollout inference with the latest trained checkpoint."""
    cfg = _ensure_config(config_path, "Rollout config")
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    _run_command([sys.executable, "src/train.py", "--config", str(cfg)], env=env)


def analyze_rollouts() -> None:
    """Summarize rollout metrics and optionally generate plots."""
    target_dir = REPO_ROOT / "kaggle_rollouts"
    rollouts_dir = REPO_ROOT / "rollouts"
    created_link = False
    if not target_dir.exists():
        print(f"[pipeline] Rollout directory missing: {target_dir}")
        return
    if not rollouts_dir.exists():
        try:
            rollouts_dir.symlink_to(target_dir)
            created_link = True
        except OSError as exc:
            print(f"[pipeline] Failed to create symlink: {exc}")
            return
    try:
        _run_command([sys.executable, "analyze_rollouts.py"])
    finally:
        if created_link and rollouts_dir.is_symlink():
            rollouts_dir.unlink()


def visualize_rollouts(html: bool = True) -> None:
    """Render rollout pickle files to HTML for quick inspection."""
    output_dir = REPO_ROOT / "kaggle_rollouts"
    pattern = sorted(output_dir.glob("rollout_ex*.pkl"))
    if not pattern:
        print("[pipeline] No rollout pickle files found to visualize.")
        return

    for pkl_path in pattern:
        output_stem = pkl_path.with_suffix("")
        html_path = output_dir / f"{output_stem.name}.html"
        cmd = [
            sys.executable,
            "visualize_rollout.py",
            str(pkl_path),
            "--output",
            str(html_path),
        ]
        if html:
            cmd.append("--html")
        _run_command(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GNS pipeline end-to-end on Kaggle.")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip installs.")
    parser.add_argument("--force-install", action="store_true", help="Reinstall deps.")
    parser.add_argument("--skip-generate", action="store_true", help="Skip dataset creation.")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training.")
    parser.add_argument("--skip-rollout", action="store_true", help="Skip rollout inference.")
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run analyze_rollouts.py after inference.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render rollout pickles to HTML (requires inference).",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DATASET_CONFIG,
        help="Path to the dataset YAML config.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=TRAIN_CONFIG,
        help="Path to the training YAML config.",
    )
    parser.add_argument(
        "--rollout-config",
        type=Path,
        default=ROLLOUT_CONFIG,
        help="Path to the rollout YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_install:
        install_dependencies(force=args.force_install)

    if not args.skip_generate:
        generate_dataset(Path(args.dataset_config))

    if not args.skip_train:
        train_model(Path(args.train_config))

    if not args.skip_rollout:
        run_rollout(Path(args.rollout_config))
        if args.run_analysis:
            analyze_rollouts()
        if args.visualize:
            visualize_rollouts()


if __name__ == "__main__":
    main()
