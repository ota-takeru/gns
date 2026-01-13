import numpy as np
import torch

from hamiltonian_sph import (
    ConservativeConfig,
    CutoffConfig,
    DissipationConfig,
    IntegratorConfig,
    ContactConfig,
    PressureConfig,
    SPHConfig,
    TermConfig,
    WallConfig,
    WallParticleConfig,
)
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
    boundary_mode = str(metadata.get("boundary_mode", "walls")).strip().lower()

    method_name = getattr(cfg, "method", "gns")
    method_options_all = getattr(cfg, "method_options", {}) or {}
    method_options = method_options_all.get(method_name, {})

    def _estimate_particle_mass(meta: dict) -> float | None:
        """メタデータから粒子質量を推定する。

        rest_density と粒子間隔が分かれば、2D: rho * dx^2 / 3D: rho * dx^3 とする。
        (public fluid データセットでは dim=2, particle_spacing が存在)
        """

        try:
            spacing = float(meta.get("particle_spacing"))
            rho0 = float(meta.get("rest_density"))
            dim = int(meta.get("dim"))
        except (TypeError, ValueError):
            return None
        if spacing <= 0 or rho0 <= 0:
            return None
        if dim == 3:
            return rho0 * (spacing ** 3)
        if dim == 2:
            return rho0 * (spacing ** 2)
        return None

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
        "boundary_mode": boundary_mode,
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

        def _sanitize_dict(val, allowed_keys: set[str], label: str):
            if val is None or not isinstance(val, dict):
                return val
            filtered = {k: v for k, v in val.items() if k in allowed_keys}
            dropped = [k for k in val if k not in allowed_keys]
            if dropped:
                print(
                    f"[simulator_factory] hamiltonian_sph.{label}: 未使用キーを無視します: {', '.join(dropped)}"
                )
            return filtered

        method_options = method_options.copy()

        # レガシー名称をマッピング
        legacy_include = method_options.pop("include_external_potential", None)
        if legacy_include is not None:
            method_options["include_external_acceleration"] = bool(legacy_include)

        method_options["sph"] = _sanitize_dict(
            method_options.get("sph"),
            allowed_keys={"smoothing_length", "max_num_neighbors"},
            label="sph",
        )
        method_options["conservative"] = _sanitize_dict(
            method_options.get("conservative"),
            allowed_keys={
                "mlp_layers",
                "mlp_hidden_dim",
                "dropout",
                "phi_max_multiplier",
                "use_density",
            },
            label="conservative",
        )
        method_options["dissipation"] = _sanitize_dict(
            method_options.get("dissipation"),
            allowed_keys={"mlp_layers", "mlp_hidden_dim", "dropout", "alpha_max", "s_clip_scaled"},
            label="dissipation",
        )
        method_options["terms"] = _sanitize_dict(
            method_options.get("terms"),
            allowed_keys={
                "enable_central",
                "enable_pressure",
                "enable_contact",
                "enable_damping",
                "w_central",
                "w_pressure",
                "w_contact",
                "w_damping",
            },
            label="terms",
        )
        method_options["pressure"] = _sanitize_dict(
            method_options.get("pressure"),
            allowed_keys={
                "enabled",
                "mlp_layers",
                "mlp_hidden_dim",
                "dropout",
                "rep_max_multiplier",
                "use_compression_gate",
            },
            label="pressure",
        )
        method_options["contact"] = _sanitize_dict(
            method_options.get("contact"),
            allowed_keys={
                "enabled",
                "mlp_layers",
                "mlp_hidden_dim",
                "dropout",
                "alpha_max",
                "s_clip_scaled",
            },
            label="contact",
        )
        method_options["wall"] = _sanitize_dict(
            method_options.get("wall"),
            allowed_keys={
                "enabled",
                "mlp_layers",
                "mlp_hidden_dim",
                "dropout",
                "a_wall_max_multiplier",
                "d0_multiplier",
                "use_velocity_gate",
            },
            label="wall",
        )
        method_options["wall_particles"] = _sanitize_dict(
            method_options.get("wall_particles"),
            allowed_keys={
                "enabled",
                "particle_type_id",
                "freeze_walls",
                "zero_wall_acceleration",
                "disable_wall_mlp",
                "disable_rollout_reflection",
            },
            label="wall_particles",
        )
        method_options["cutoff"] = _sanitize_dict(
            method_options.get("cutoff"), allowed_keys={"kind"}, label="cutoff"
        )
        method_options["integrator"] = _sanitize_dict(
            method_options.get("integrator"),
            allowed_keys={"dt", "type", "dt_source"},
            label="integrator",
        )

        dt_source = None
        integrator_opts = method_options.get("integrator")
        if isinstance(integrator_opts, dict):
            dt_source = integrator_opts.pop("dt_source", None)
        if dt_source is not None:
            dt_source = str(dt_source).strip().lower()
            if dt_source not in {"auto", "metadata", "config"}:
                print(
                    "[simulator_factory] integrator.dt_source は auto/metadata/config のみ対応です。"
                )
                dt_source = None

        method_options["sph"] = _maybe(SPHConfig, "sph")
        method_options["conservative"] = _maybe(ConservativeConfig, "conservative")
        method_options["dissipation"] = _maybe(DissipationConfig, "dissipation")
        method_options["terms"] = _maybe(TermConfig, "terms")
        method_options["pressure"] = _maybe(PressureConfig, "pressure")
        method_options["contact"] = _maybe(ContactConfig, "contact")
        method_options["wall"] = _maybe(WallConfig, "wall")
        method_options["wall_particles"] = _maybe(WallParticleConfig, "wall_particles")
        method_options["cutoff"] = _maybe(CutoffConfig, "cutoff")
        method_options["integrator"] = _maybe(IntegratorConfig, "integrator")

        allowed_keys = {
            "sph",
            "conservative",
            "dissipation",
            "terms",
            "pressure",
            "contact",
            "wall",
            "wall_particles",
            "cutoff",
            "integrator",
            "particle_mass",
            "gravity",
            "pos_feature_scale",
            "vel_feature_scale",
            "include_external_acceleration",
        }
        unused_keys = [k for k in list(method_options.keys()) if k not in allowed_keys]
        for key in unused_keys:
            method_options.pop(key, None)
            print(f"[simulator_factory] hamiltonian_sph: 未使用キーを無視します: {key}")

        # データセット側の時間刻み dt がメタデータに入っていれば、積分器に反映する。
        # （デフォルトの 1e-3 のままだと実データの 0.006 などと大きくずれて速度推定が跳ね、
        #  ロールアウトで粒子が不安定になりやすい）
        dt_from_meta = metadata.get("dt")
        if dt_from_meta is not None:
            dt_from_meta = float(dt_from_meta)
            integrator_cfg = method_options["integrator"] or IntegratorConfig()
            current_dt = getattr(integrator_cfg, "dt", None)
            dt_source_mode = dt_source or "auto"
            if dt_source_mode == "config":
                pass
            elif dt_source_mode == "metadata":
                integrator_cfg.dt = dt_from_meta
            else:
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

        # 粒子質量が未指定ならメタデータから自動推定する。
        if "particle_mass" not in method_options or method_options.get("particle_mass") is None:
            estimated_mass = _estimate_particle_mass(metadata)
            if estimated_mass is not None:
                method_options["particle_mass"] = estimated_mass
                print(
                    f"[simulator_factory] particle_mass をメタデータから推定: {estimated_mass:.4g}"
                )

    SimulatorClass = get_simulator_class(method_name)
    simulator = SimulatorClass(**(base_kwargs | method_options))
    return simulator


__all__ = ["_get_simulator"]
