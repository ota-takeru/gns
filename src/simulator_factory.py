import numpy as np
import torch

from hamiltonian_sph import HamiltonianNetConfig, IntegratorConfig, SPHConfig
from learned_simulator import BaseSimulator, get_simulator_class
from train_config import Config, NUM_PARTICLE_TYPES


def _get_simulator(
    metadata: dict,
    acc_noise_std: float,
    vel_noise_std: float,
    device: torch.device,
    cfg: Config,
) -> BaseSimulator:
    """メタ情報と設定から Simulator を生成"""
    normalization_stats = {
        "acceleration": {
            "mean": torch.FloatTensor(metadata["acc_mean"]).to(device),
            "std": torch.sqrt(
                torch.FloatTensor(metadata["acc_std"]) ** 2 + acc_noise_std**2
            ).to(device),
        },
        "velocity": {
            "mean": torch.FloatTensor(metadata["vel_mean"]).to(device),
            "std": torch.sqrt(
                torch.FloatTensor(metadata["vel_std"]) ** 2 + vel_noise_std**2
            ).to(device),
        },
    }

    if "nnode_in" in metadata and "nedge_in" in metadata:
        nnode_in = metadata["nnode_in"]
        nedge_in = metadata["nedge_in"]
    else:
        nnode_in = 37 if metadata["dim"] == 3 else 30
        nedge_in = metadata["dim"] + 1

    if "bounds" not in metadata:
        msg = (
            "Dataset metadata is missing 'bounds'. "
            "Regenerate the dataset with wall-distance features."
        )
        raise KeyError(msg)
    boundaries = np.asarray(metadata["bounds"], dtype=np.float32)
    if boundaries.ndim != 2 or boundaries.shape[1] != 2:
        msg = (
            f"Invalid bounds shape {boundaries.shape}, expected (dim, 2). "
            "Check dataset generation."
        )
        raise ValueError(msg)
    boundary_clamp_limit = float(metadata.get("boundary_augment", 1.0))

    method_name = getattr(cfg, "method", "gns")
    method_options_all = getattr(cfg, "method_options", {}) or {}
    method_options = method_options_all.get(method_name, {})

    base_kwargs = {
        "particle_dimensions": metadata["dim"],
        "nnode_in": nnode_in,
        "nedge_in": nedge_in,
        "latent_dim": 128,
        "nmessage_passing_steps": 10,
        "nmlp_layers": 2,
        "mlp_hidden_dim": 128,
        "connectivity_radius": metadata["default_connectivity_radius"],
        "normalization_stats": normalization_stats,
        "nparticle_types": NUM_PARTICLE_TYPES,
        "particle_type_embedding_size": 16,
        "boundaries": boundaries,
        "boundary_clamp_limit": boundary_clamp_limit,
        "device": device,
    }

    if method_name == "hamiltonian_sph":
        # YAML のネスト dict → dataclass に変換（未指定ならデフォルト）
        def _maybe(cls, key):
            val = method_options.get(key)
            if val is None:
                return None
            if isinstance(val, cls):
                return val
            if isinstance(val, dict):
                return cls(**val)
            raise TypeError(f"{key} must be dict or {cls.__name__}")

        method_options = method_options.copy()
        method_options["sph"] = _maybe(SPHConfig, "sph")
        method_options["hamiltonian_net"] = _maybe(HamiltonianNetConfig, "hamiltonian_net")
        method_options["integrator"] = _maybe(IntegratorConfig, "integrator")

        # データセット側の時間刻み dt がメタデータに入っていれば、積分器に反映する。
        # （デフォルトの 1e-3 のままだと実データの 0.006 などと大きくずれて速度推定が跳ね、
        #  ロールアウトで粒子が不安定になりやすい）
        dt_from_meta = metadata.get("dt")
        if dt_from_meta is not None:
            dt_from_meta = float(dt_from_meta)
            integrator_cfg = method_options["integrator"] or IntegratorConfig()
            current_dt = getattr(integrator_cfg, "dt", None)
            # ユーザが明示指定していない、もしくは大きく乖離している場合は上書きする
            if current_dt is None or current_dt <= 0:
                integrator_cfg.dt = dt_from_meta
            elif abs(current_dt - dt_from_meta) / max(dt_from_meta, 1e-9) > 0.05:
                print(
                    f"[simulator_factory] integrator.dt({current_dt}) を "
                    f"データセットの dt({dt_from_meta}) に合わせて上書きします。"
                )
                integrator_cfg.dt = dt_from_meta
            method_options["integrator"] = integrator_cfg

    SimulatorClass = get_simulator_class(method_name)
    simulator = SimulatorClass(**(base_kwargs | method_options))
    return simulator


__all__ = ["_get_simulator"]
