import inspect
import math
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from learned_simulator import BaseSimulator
from train_config import Config

# TensorBoard はオプション依存。ここでは遅延 import にして、
# cfg.tensorboard_enable=False の場合に ImportError ログが出ないようにする。
tensorboard_program: Any | None = None

try:  # Prefer the new torch.amp API when available to avoid deprecation warnings.
    from torch.amp import autocast, GradScaler  # type: ignore[attr-defined]
    _AMP_SOURCE = "torch.amp"
except (ImportError, AttributeError):  # pragma: no cover - depends on torch version
    from torch.cuda.amp import autocast, GradScaler  # type: ignore[no-redef]
    _AMP_SOURCE = "torch.cuda.amp"

try:
    _AMP_AUTOCAST_SUPPORTS_DEVICE_TYPE = (
        "device_type" in inspect.signature(autocast).parameters
    )
except (ValueError, TypeError):
    _AMP_AUTOCAST_SUPPORTS_DEVICE_TYPE = False

try:
    _AMP_GRADSCALER_SUPPORTS_DEVICE_TYPE = (
        "device_type" in inspect.signature(GradScaler).parameters
    )
except (ValueError, TypeError):
    _AMP_GRADSCALER_SUPPORTS_DEVICE_TYPE = False


def optimizer_to(optim: torch.optim.Optimizer, device: torch.device):
    """オプティマイザ内部のテンソルを所定デバイスへ移動"""
    for state in optim.state.values():
        if isinstance(state, torch.Tensor):
            state.data = state.data.to(device)
            if state._grad is not None:
                state._grad.data = state._grad.data.to(device)
        elif isinstance(state, dict):
            for sub in state.values():
                if isinstance(sub, torch.Tensor):
                    sub.data = sub.data.to(device)
                    if sub._grad is not None:
                        sub._grad.data = sub._grad.data.to(device)


def save_model_and_train_state(
    simulator: BaseSimulator | DDP,
    cfg: Config,
    step: int,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    train_loss: float | None,
    valid_loss: float | None,
    train_loss_hist: list[tuple[int, float]],
    valid_loss_hist: list[tuple[int, float]],
):
    Path(cfg.model_path).mkdir(parents=True, exist_ok=True)
    model_to_save = simulator.module if isinstance(simulator, DDP) else simulator
    model_fname = Path(cfg.model_path) / f"model-{step}.pt"
    model_to_save.save(str(model_fname))

    train_state = dict(
        optimizer_state=optimizer.state_dict(),
        global_train_state={
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        },
        loss_history={"train": train_loss_hist, "valid": valid_loss_hist},
        used_config=_asjsonable_cfg(cfg),
    )
    ts_fname = Path(cfg.model_path) / f"train_state-{step}.pt"
    torch.save(train_state, ts_fname)


def _asjsonable_cfg(cfg: Config) -> dict[str, Any]:
    return {
        "mode": cfg.mode,
        "data_path": cfg.data_path,
        "scenario": cfg.scenario,
        "scenario_options": cfg.scenario_options,
        "active_scenario": (
            {
                "key": cfg.active_scenario.key,
                "dataset_dir": str(cfg.active_scenario.dataset_dir),
            }
            if cfg.active_scenario
            else None
        ),
        "model_path": cfg.model_path,
        "output_path": cfg.output_path,
        "output_filename": cfg.output_filename,
        "batch_size": cfg.batch_size,
        "noise_std": cfg.noise_std,
        "log_interval": cfg.log_interval,
        "max_grad_norm": cfg.max_grad_norm,
        "ntraining_steps": cfg.ntraining_steps,
        "validation_interval": cfg.validation_interval,
        "nsave_steps": cfg.nsave_steps,
        "rollout_interval": cfg.rollout_interval,
        "rollout_max_examples": cfg.rollout_max_examples,
        "rollout_dataset": cfg.rollout_dataset,
        "rollout_inference_max_examples": cfg.rollout_inference_max_examples,
        "train_dataset_fraction": cfg.train_dataset_fraction,
        "valid_dataset_fraction": cfg.valid_dataset_fraction,
        "train_dataset_count": cfg.train_dataset_count,
        "valid_dataset_count": cfg.valid_dataset_count,
        "tensorboard_enable": cfg.tensorboard_enable,
        "tensorboard_interval": cfg.tensorboard_interval,
        "lr_init": cfg.lr_init,
        "lr_decay": cfg.lr_decay,
        "lr_decay_steps": cfg.lr_decay_steps,
        "model_file": cfg.model_file,
        "train_state_file": cfg.train_state_file,
        "cuda_device_number": cfg.cuda_device_number,
        "tensorboard_log_dir": cfg.tensorboard_log_dir,
        "tensorboard_port": cfg.tensorboard_port,
        "tensorboard_host": cfg.tensorboard_host,
        "seed": cfg.seed,
        "enable_ddp": cfg.enable_ddp,
        "ddp_backend": cfg.ddp_backend,
        "ddp_timeout_sec": cfg.ddp_timeout_sec,
        "ddp_find_unused_parameters": cfg.ddp_find_unused_parameters,
        "amp_enable": cfg.amp_enable,
        "amp_dtype": cfg.amp_dtype,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.persistent_workers,
        "pin_memory": cfg.pin_memory,
        "prefetch_factor": cfg.prefetch_factor,
        "world_size": cfg.world_size,
        "rank": cfg.rank,
        "local_rank": cfg.local_rank,
    }


def _format_eta(seconds: float) -> str:
    sec = int(max(0, seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _resolve_amp_dtype(name: str) -> torch.dtype:
    normalized = name.lower().strip()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        if not torch.cuda.is_available():
            raise ValueError(
                "bfloat16 is only supported on CUDA/CPU with appropriate hardware."
            )
        return torch.bfloat16
    if normalized in {"float32", "fp32", "single"}:
        return torch.float32
    msg = f"Unsupported amp_dtype '{name}'."
    raise ValueError(msg)


def _is_distributed_env(cfg: Config) -> bool:
    if not cfg.enable_ddp or not dist.is_available():
        return False
    required_env = {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
    if not required_env.issubset(os.environ.keys()):
        return False
    # torchrun --nproc_per_node=1 などで WORLD_SIZE=1 の場合は DDP を無効化して余計なオーバーヘッドを避ける
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    return world_size > 1


def _init_distributed(cfg: Config) -> tuple[bool, int, int, int]:
    if not _is_distributed_env(cfg):
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if dist.is_initialized():
        return True, rank, world_size, local_rank

    timeout = timedelta(seconds=cfg.ddp_timeout_sec)
    dist.init_process_group(
        backend=cfg.ddp_backend,
        timeout=timeout,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _unwrap_simulator(
    simulator: BaseSimulator | DDP,
) -> BaseSimulator:
    return simulator.module if isinstance(simulator, DDP) else simulator


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _launch_tensorboard(
    log_dir: Path, host: str | None, port: int | None
) -> tuple[Any | None, str | None]:
    global tensorboard_program
    # 必要になった時だけ import する（tensorboard が未インストールでも静かにスキップ）
    if tensorboard_program is None:
        try:
            from tensorboard import program as tensorboard_program  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            return None, None
    if tensorboard_program is None:
        return None, None
    tb = tensorboard_program.TensorBoard()
    argv = [None, "--logdir", str(log_dir)]
    if host:
        argv += ["--host", host]
    if port:
        argv += ["--port", str(port)]
    try:
        tb.configure(argv=argv)
        url = tb.launch()
    except Exception:  # pragma: no cover - runtime environment dependent
        return None, None
    return tb, url


def _compute_grad_norm(parameters) -> float | None:
    total = 0.0
    has_grad = False
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
        has_grad = True
    if not has_grad:
        return None
    return math.sqrt(total)


def _resolve_rollout_dataset_path(cfg: Config) -> Path | None:
    base = Path(cfg.data_path)
    if cfg.rollout_dataset:
        candidate = base / f"{cfg.rollout_dataset}.npz"
        if candidate.exists():
            return candidate
    scenario = cfg.active_scenario
    if scenario and scenario.rollout_dataset:
        candidate = base / f"{scenario.rollout_dataset}.npz"
        if candidate.exists():
            return candidate
    for split_name in ("valid", "test", "train"):
        candidate = base / f"{split_name}.npz"
        if candidate.exists():
            return candidate
    return None


__all__ = [
    "optimizer_to",
    "save_model_and_train_state",
    "_asjsonable_cfg",
    "_format_eta",
    "_resolve_amp_dtype",
    "_is_distributed_env",
    "_init_distributed",
    "_cleanup_distributed",
    "_unwrap_simulator",
    "_set_seed",
    "_launch_tensorboard",
    "_compute_grad_norm",
    "_resolve_rollout_dataset_path",
    "autocast",
    "GradScaler",
    "_AMP_AUTOCAST_SUPPORTS_DEVICE_TYPE",
    "_AMP_GRADSCALER_SUPPORTS_DEVICE_TYPE",
]
