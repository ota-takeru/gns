from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# -----------------------
# 定数
# -----------------------
INPUT_SEQUENCE_LENGTH = 6  # 先頭6ステップを履歴として使う(速度5本が引ける)
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3


@dataclass
class Config:
    mode: str = "train"  # train / valid / rollout
    data_path: str = "./datasets/"
    scenario: str = "fluid"
    scenario_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    method: str = "gns"  # 使用する手法識別子
    method_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_scenario: Any | None = field(default=None, repr=False)
    model_path: str = "./models/"
    output_path: str = "./rollouts/"
    output_filename: str = "rollout"

    batch_size: int = 2
    noise_std: float = 6.7e-4
    noise: str = "random_walk"
    loss: str = "acceleration"
    log_interval: int = 10
    max_grad_norm: float | None = 1.0

    ntraining_steps: int = int(2e7)
    validation_interval: int | None = None
    nsave_steps: int = 5000
    rollout_interval: int | None = None
    rollout_max_examples: int = 1
    rollout_dataset: str | None = None
    # rollout 指標を計算するときの最大ステップ数(None なら軌道全体)
    rollout_max_steps: int | None = None
    # rollout(推論)時に保存するシーン数の上限(Noneで全件)
    rollout_inference_max_examples: int | None = None
    train_dataset_fraction: float | None = None
    valid_dataset_fraction: float | None = None
    train_dataset_count: int | None = None
    valid_dataset_count: int | None = None
    tensorboard_enable: bool = False
    tensorboard_interval: int | None = None

    lr_init: float = 1e-4
    lr_decay: float = 0.1
    lr_decay_steps: int = int(5e6)
    warmup_steps: int | None = 2000

    model_file: str | None = None  # "latest" (inference default) or ファイル名 or null
    train_state_file: str | None = "train_state.pt"  # "latest" or ファイル名 or null

    cuda_device_number: int | None = None  # null で自動選択
    tensorboard_log_dir: str | None = None
    tensorboard_port: int | None = None
    tensorboard_host: str | None = None
    log_file: str | None = None  # null なら model_path/train.log に追記
    seed: int = 42
    enable_ddp: bool = True
    ddp_backend: str = "nccl"
    ddp_timeout_sec: int = 1800
    ddp_find_unused_parameters: bool = False
    ddp_async_error_handling: bool = False
    ddp_torch_distributed_debug: str | None = None  # "INFO" / "DETAIL" / None
    amp_enable: bool = False
    amp_dtype: str = "float16"
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int | None = None
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0


def load_config(path: str) -> Config:
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
