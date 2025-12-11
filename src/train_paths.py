import re
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from train_config import Config


def _prepare_model_directory(cfg: Config) -> None:
    """モデル保存基底ディレクトリを作成し、パスを記録"""
    base_dir = Path(cfg.model_path).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    setattr(cfg, "model_base_path", str(base_dir))


def _resolve_output_directory(cfg: Config) -> Path:
    """rollout/valid 出力を method/run_id 配下にまとめる"""
    base_dir = Path(cfg.output_path).expanduser()
    run_dir = base_dir / cfg.method / cfg.output_filename
    run_dir.mkdir(parents=True, exist_ok=True)
    setattr(cfg, "resolved_output_dir", str(run_dir))
    return run_dir


def _resolve_model_run_directory(cfg: Config) -> None:
    """学習時の model_path を run ごとのディレクトリに切る"""
    base_dir = Path(getattr(cfg, "model_base_path", cfg.model_path)).expanduser().resolve()
    needs_run_dir = cfg.mode == "train" and cfg.model_file is None
    distributed = (
        getattr(cfg, "world_size", 1) > 1
        and dist.is_available()
        and dist.is_initialized()
    )

    if not needs_run_dir:
        setattr(cfg, "model_run_path", None)
        cfg.model_path = str(base_dir)
        return

    run_dir: Optional[Path] = None
    if cfg.rank == 0:
        timestamp = time.strftime("run-%Y%m%d-%H%M%S")
        run_dir = base_dir / timestamp
        counter = 1
        while run_dir.exists():
            run_dir = base_dir / f"{timestamp}-{counter:02d}"
            counter += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        run_dir = run_dir.resolve()

    if distributed:
        payload = [str(run_dir) if run_dir is not None else ""]
        dist.broadcast_object_list(payload, src=0)
        if not payload[0]:
            msg = "Failed to broadcast model run directory from rank 0."
            raise RuntimeError(msg)
        run_dir = Path(payload[0]).resolve()
        dist.barrier()
    elif run_dir is None:
        run_dir = base_dir.resolve()

    setattr(cfg, "model_run_path", str(run_dir))
    cfg.model_path = str(run_dir)


def _resolve_model_path(cfg: Config) -> str:
    """cfg.model_file が 'latest' のとき最新を解決。そうでなければ結合して返す。"""
    Path(cfg.model_path).mkdir(parents=True, exist_ok=True)
    model_file = cfg.model_file
    if model_file is None and cfg.mode in ("valid", "rollout"):
        model_file = "latest"

    base_dir = (
        Path(getattr(cfg, "model_base_path", cfg.model_path)).expanduser().resolve()
    )
    search_root = base_dir if base_dir.is_dir() else base_dir.parent

    if model_file == "latest":
        expr = re.compile(r"model-(\d+)\.pt$")
        candidates: list[tuple[float, int, Path]] = []
        for path in search_root.rglob("model-*.pt"):
            match = expr.search(path.name)
            if not match:
                continue
            try:
                step_num = int(match.group(1))
            except ValueError:
                step_num = -1
            stat = path.stat()
            candidates.append((stat.st_mtime, step_num, path.resolve()))
        if not candidates:
            msg = f"No model files found in {search_root}"
            raise FileNotFoundError(msg)
        candidates.sort()
        return str(candidates[-1][2])
    if model_file:
        candidate = Path(model_file)
        if candidate.is_file():
            return str(candidate.resolve())

        search_roots = [Path(cfg.model_path).expanduser().resolve()]
        if base_dir not in search_roots:
            search_roots.append(base_dir)

        for root in search_roots:
            resolved = (root / model_file).expanduser().resolve()
            if resolved.exists():
                return str(resolved)

        msg = f"Model file {model_file} not found under {', '.join(str(r) for r in search_roots)}"
        raise FileNotFoundError(msg)
    return str(Path(cfg.model_path) / "model-init.pt")


__all__ = [
    "_prepare_model_directory",
    "_resolve_output_directory",
    "_resolve_model_run_directory",
    "_resolve_model_path",
]
