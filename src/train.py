import argparse
import json
from pathlib import Path

import torch

from predictor import predict
from scenarios import ScenarioRegistry
from train_config import Config, load_config
from train_paths import _prepare_model_directory, _resolve_model_run_directory
from train_utils import _asjsonable_cfg, _cleanup_distributed, _init_distributed
from trainer import train


def _print_config(cfg: Config, resolved_cfg_path: Path):
    if cfg.rank != 0:
        return
    print(f"config: {resolved_cfg_path}")
    print(f"  mode={cfg.mode} data_path={cfg.data_path} scenario={cfg.scenario}")
    if cfg.scenario_options:
        option_keys = ", ".join(sorted(cfg.scenario_options))
        print(f"  scenario_options: {option_keys}")
    else:
        print("  scenario_options: (none)")


def _prepare_scenario(cfg: Config) -> Config:
    base_dataset = Path(cfg.data_path).expanduser().resolve()
    default_aliases = [cfg.scenario]
    if cfg.scenario != "rigid":
        default_aliases.append("rigid")
    registry = ScenarioRegistry(
        base_dataset, cfg.scenario_options, default_keys=default_aliases
    )
    scenario = registry.get(cfg.scenario)
    scenario.apply_overrides(cfg)
    cfg.data_path = str(scenario.dataset_dir)
    if cfg.rollout_dataset is None and scenario.rollout_dataset:
        cfg.rollout_dataset = scenario.rollout_dataset
    cfg.active_scenario = scenario
    if cfg.rank == 0:
        print(f"scenario: {scenario.key} @ {scenario.dataset_dir}")
        if scenario.description:
            print(f"  description: {scenario.description}")
    return cfg


def _select_device(cfg: Config, distributed: bool, local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        if distributed:
            return torch.device(f"cuda:{local_rank}")
        if cfg.cuda_device_number is not None:
            return torch.device(f"cuda:{int(cfg.cuda_device_number)}")
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser(description="GNS Runner")
    p.add_argument("--config", "-c", default="config.yaml", help="Path to YAML config")
    args = p.parse_args()

    cfg = load_config(args.config)
    distributed, rank, world_size, local_rank = _init_distributed(cfg)
    cfg.world_size = world_size
    cfg.rank = rank
    cfg.local_rank = local_rank

    try:
        resolved_cfg_path = Path(args.config).expanduser()
        if not resolved_cfg_path.is_absolute():
            resolved_cfg_path = (Path.cwd() / resolved_cfg_path).resolve()
        else:
            resolved_cfg_path = resolved_cfg_path.resolve()
    except Exception:
        resolved_cfg_path = Path(args.config)

    _print_config(cfg, resolved_cfg_path)
    cfg = _prepare_scenario(cfg)

    _prepare_model_directory(cfg)
    _resolve_model_run_directory(cfg)
    if cfg.rank == 0 and getattr(cfg, "model_run_path", None):
        print(f"model run directory: {cfg.model_run_path}")

    device = _select_device(cfg, distributed, local_rank)
    if cfg.rank == 0:
        print(f"device: {device}")

    Path(cfg.model_path).mkdir(parents=True, exist_ok=True)
    with Path.open(
        Path(cfg.model_path) / "used_config.json", "w", encoding="utf-8"
    ) as f:
        json.dump(_asjsonable_cfg(cfg), f, ensure_ascii=False, indent=2)

    if cfg.mode == "train":
        train(cfg, device)
    elif cfg.mode in ("valid", "rollout"):
        predict(cfg, device)
    else:
        msg = f"Unknown mode: {cfg.mode}"
        raise ValueError(msg)

    if distributed:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
