import argparse
import inspect
import json
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

try:
    from . import dataset_utils
except ImportError:  # pragma: no cover
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = CURRENT_DIR.parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from datasets.scripts import dataset_utils

KINEMATIC_PARTICLE_ID = 3
BOUNDARY_MODE_WALLS = "walls"
BOUNDARY_MODE_PERIODIC = "periodic"


@dataclass
class DomainConfig:
    left: float = -5.0
    right: float = 5.0
    bottom: float = 0.0
    top: float = 6.0

    def as_array(self) -> np.ndarray:
        return np.array([[self.left, self.right], [self.bottom, self.top]], dtype=np.float32)


@dataclass
class EmitterConfig:
    origin: tuple[float, float] = (-4.5, 0.2)
    size: tuple[float, float] = (3.0, 3.0)
    jitter: float = 0.01
    initial_velocity: tuple[float, float] = (0.0, 0.0)
    random_velocity_scale: float = 0.5
    fill_ratio_range: tuple[float, float] = (0.9, 1.1)


@dataclass
class ObstacleConfig:
    center: tuple[float, float]
    size: tuple[float, float]
    padding: float = 0.0
    damping: Optional[float] = None
    layers: Optional[int] = None  # YAML 互換用（粒子障害物用レイヤー、ここでは未使用）

    def bounds(self) -> tuple[float, float, float, float]:
        cx, cy = self.center
        sx, sy = self.size
        if sx <= 0 or sy <= 0:
            raise ValueError("Obstacle size must be positive in both axes.")
        half_w = sx * 0.5
        half_h = sy * 0.5
        return (cx - half_w, cx + half_w, cy - half_h, cy + half_h)


@dataclass
class TankMotionConfig:
    """Sinusoidal加速度でタンクを揺する設定（スロッシング用）。"""

    axis: str = "x"
    amplitude: float = 4.0  # m/s^2 揺動の強さ
    frequency: float = 0.8  # Hz
    phase: float = 0.0  # rad
    ramp_time: float = 0.2  # s 立ち上がり時間
    offset: float = 0.0  # 定常加速度

    def axis_index(self) -> int:
        ax = self.axis.lower()
        if ax not in {"x", "y"}:
            raise ValueError("tank_motion.axis must be 'x' or 'y'.")
        return 0 if ax == "x" else 1

    def acceleration(self, t: float) -> float:
        omega = 2.0 * math.pi * self.frequency
        base = self.amplitude * math.sin(omega * t + self.phase) + self.offset
        if self.ramp_time > 0:
            ramp = min(1.0, max(0.0, t / self.ramp_time))
            return base * ramp
        return base


@dataclass
class FluidCaseConfig:
    name: str
    output_subdir: str | None = None
    num_train_scenes: int = 200
    num_valid_scenes: int = 40
    timesteps: int = 240
    dt: float = 0.006
    solver_substeps: int = 2
    particle_spacing: float = 0.12
    kernel_radius_scale: float = 2.0
    rest_density: float = 1000.0
    stiffness: float = 2000.0
    sound_speed: float = 40.0
    viscosity: float = 40.0
    boundary_damping: float = 0.5
    boundary_clamp_limit: float = 1.0
    wall_particle_layers: int = 0
    gravity: tuple[float, float] = field(default_factory=lambda: (0.0, -9.81))
    boundary_mode: str = BOUNDARY_MODE_WALLS
    xsph_factor: float = 0.0
    position_noise: float = 0.01
    seed: int = 42
    domain: DomainConfig = field(default_factory=DomainConfig)
    emitter: EmitterConfig = field(default_factory=EmitterConfig)
    obstacles: list[ObstacleConfig] = field(default_factory=list)
    tank_motion: Optional[TankMotionConfig] = None

    def resolve_output_dir(self, root: Path) -> Path:
        return root / (self.output_subdir or self.name)


@dataclass
class FluidDatasetConfig:
    output_root: Path
    cases: list[FluidCaseConfig]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping.")
    return data


def _merge_dict(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(default)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _load_inline_case(value: str) -> dict[str, Any]:
    """Load a single case mapping from a YAML/JSON string or file path."""
    path_candidate = Path(value)
    try:
        if path_candidate.exists():
            return _load_yaml(path_candidate)
    except OSError:
        # Treat overly long or invalid paths as inline YAML/JSON strings
        pass
    data = yaml.safe_load(value)
    if not isinstance(data, dict):
        raise ValueError("Inline case must be a mapping (YAML/JSON object).")
    return data


def _build_dataset_config(raw: dict[str, Any]) -> FluidDatasetConfig:
    defaults = raw.get("defaults", {})
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, Iterable) or isinstance(cases_raw, dict):
        raise ValueError("`cases` must be a list of case configurations.")

    output_root = Path(raw.get("output_root", "./datasets/out_fluid")).resolve()
    cases: list[FluidCaseConfig] = []
    for entry in cases_raw:
        if not isinstance(entry, dict):
            raise ValueError("Each case entry must be a mapping.")
        merged = _merge_dict(defaults, entry)
        domain_cfg = DomainConfig(**merged.get("domain", {}))
        emitter_cfg = EmitterConfig(**merged.get("emitter", {}))
        obstacles_cfg: list[ObstacleConfig] = []
        raw_obstacles = merged.get("obstacles", [])
        if isinstance(raw_obstacles, dict):
            raise ValueError("`obstacles` must be a list of obstacle configurations.")
        if raw_obstacles:
            if not isinstance(raw_obstacles, Iterable):
                raise ValueError("`obstacles` must be iterable when provided.")
            for idx, obs in enumerate(raw_obstacles):
                if not isinstance(obs, dict):
                    raise ValueError(f"Obstacle #{idx} must be a mapping.")
                try:
                    obstacles_cfg.append(ObstacleConfig(**obs))
                except TypeError as exc:  # pragma: no cover - defensive
                    raise ValueError(f"Invalid obstacle #{idx}: {exc}") from exc
                # Validate size positivity early
                obstacles_cfg[-1].bounds()
        tank_motion_cfg: Optional[TankMotionConfig] = None
        raw_tank_motion = merged.get("tank_motion")
        if raw_tank_motion is not None:
            if not isinstance(raw_tank_motion, dict):
                raise ValueError("`tank_motion` must be a mapping when provided.")
            tank_motion_cfg = TankMotionConfig(**raw_tank_motion)
        case_kwargs = {
            key: value
            for key, value in merged.items()
            if key not in {"domain", "emitter", "obstacles", "tank_motion"}
        }
        case = FluidCaseConfig(
            domain=domain_cfg,
            emitter=emitter_cfg,
            obstacles=obstacles_cfg,
            tank_motion=tank_motion_cfg,
            **case_kwargs,
        )
        cases.append(case)

    if not cases:
        raise ValueError("At least one case must be specified.")
    return FluidDatasetConfig(output_root=output_root, cases=cases)


def load_config(path: Path) -> FluidDatasetConfig:
    raw = _load_yaml(path)
    return _build_dataset_config(raw)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_boundary_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in {BOUNDARY_MODE_WALLS, BOUNDARY_MODE_PERIODIC}:
        raise ValueError(
            f"Unsupported boundary_mode '{mode}'. Use '{BOUNDARY_MODE_WALLS}' or '{BOUNDARY_MODE_PERIODIC}'."
        )
    return value


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs
    return {key: val for key, val in kwargs.items() if key in sig.parameters}


try:
    from pysph.base.nnps import DomainManager, LinkedListNNPS
    from pysph.base.utils import get_particle_array_wcsph
    from pysph.sph.scheme import WCSPHScheme
except ImportError:  # pragma: no cover - PySPH must be available at runtime.
    DomainManager = None  # type: ignore[assignment]
    LinkedListNNPS = None  # type: ignore[assignment]
    get_particle_array_wcsph = None  # type: ignore[assignment]
    WCSPHScheme = None  # type: ignore[assignment]


class PySPHSimulation:
    """Dataset-oriented 2D fluid simulation driven by PySPH."""

    def __init__(self, cfg: FluidCaseConfig, rng: random.Random) -> None:
        if WCSPHScheme is None or get_particle_array_wcsph is None:
            raise RuntimeError(
                "PySPH is required for fluid dataset generation. "
                "Install it with `pip install pysph`."
            )

        self.cfg = cfg
        self.rng = rng
        self.spacing = cfg.particle_spacing
        self.h = max(cfg.kernel_radius_scale * self.spacing, 1e-4)
        self.mass = cfg.rest_density * (self.spacing**2)
        self.substeps = max(1, int(cfg.solver_substeps))
        self.dt = cfg.dt
        self.gravity = np.array(cfg.gravity, dtype=np.float32)
        self.boundary_mode = _normalize_boundary_mode(cfg.boundary_mode)
        self.domain_min = np.array([cfg.domain.left, cfg.domain.bottom], dtype=np.float32)
        self.domain_max = np.array([cfg.domain.right, cfg.domain.top], dtype=np.float32)
        self.boundary_margin = (
            0.0 if self.boundary_mode == BOUNDARY_MODE_PERIODIC else self.spacing * 0.5
        )
        self.obstacles = cfg.obstacles
        self.tank_motion = cfg.tank_motion
        self._tank_axis_idx = (
            None if self.tank_motion is None else int(self.tank_motion.axis_index())
        )
        (
            self._obs_bounds,
            self._obs_margins,
            self._obs_damping,
        ) = self._prepare_obstacles()
        self.domain_manager = self._build_domain_manager()

        self.positions0, self.velocities0 = self._initialize_particles()
        self._steps_per_frame = max(1, self.substeps)
        self._solver_dt = self.dt / self._steps_per_frame
        (
            self.fluid,
            self.boundary,
            self.scheme,
            self.solver,
            self.nnps,
        ) = self._setup_solver()

        self.history: list[np.ndarray] = []
        self._substep_counter = 0
        self._target_frames = cfg.timesteps

    def _prepare_obstacles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precompute obstacle bounds, margins, and damping for fast collision handling."""
        if not self.obstacles:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        bounds_list: list[tuple[float, float, float, float]] = []
        margins: list[float] = []
        damping: list[float] = []
        for obs in self.obstacles:
            xmin, xmax, ymin, ymax = obs.bounds()
            bounds_list.append((xmin, xmax, ymin, ymax))
            # padding は衝突後の押し出し距離に加算して数値振動を抑える
            margins.append(max(0.0, obs.padding) + self.spacing * 0.5)
            damping.append(self.cfg.boundary_damping if obs.damping is None else obs.damping)
        return (
            np.asarray(bounds_list, dtype=np.float32),
            np.asarray(margins, dtype=np.float32),
            np.asarray(damping, dtype=np.float32),
        )

    def _initialize_particles(self) -> tuple[np.ndarray, np.ndarray]:
        emit = self.cfg.emitter
        ox, oy = emit.origin
        sx, sy = emit.size
        ratio = self.rng.uniform(*emit.fill_ratio_range)
        effective_width = sx * math.sqrt(ratio)
        effective_height = sy * math.sqrt(ratio)
        nx = max(1, int(effective_width / self.spacing))
        ny = max(1, int(effective_height / self.spacing))

        positions = np.zeros((nx * ny, 2), dtype=np.float32)
        for ix in range(nx):
            for iy in range(ny):
                idx = ix * ny + iy
                positions[idx, 0] = ox + (ix + 0.5) * self.spacing
                positions[idx, 1] = oy + (iy + 0.5) * self.spacing

        if self.cfg.position_noise > 0:
            noise_scale = self.cfg.position_noise * self.spacing
            jitter = np.array(
                [
                    [
                        self.rng.uniform(-noise_scale, noise_scale),
                        self.rng.uniform(-noise_scale, noise_scale),
                    ]
                    for _ in range(len(positions))
                ],
                dtype=np.float32,
            )
            positions += jitter

        if emit.jitter > 0:
            offset = np.array(
                [
                    self.rng.uniform(-emit.jitter, emit.jitter),
                    self.rng.uniform(-emit.jitter, emit.jitter),
                ],
                dtype=np.float32,
            )
            positions += offset

        permutation = self.rng.sample(range(len(positions)), len(positions))
        positions = positions[np.array(permutation)]
        positions = np.clip(
            positions,
            self.domain_min + self.boundary_margin,
            self.domain_max - self.boundary_margin,
        )

        base_velocity = np.array(emit.initial_velocity, dtype=np.float32)
        velocities = np.repeat(base_velocity[None, :], len(positions), axis=0)
        if emit.random_velocity_scale > 0:
            random_vel = (
                np.array(
                    [
                        self.rng.uniform(-1.0, 1.0),
                        self.rng.uniform(-1.0, 1.0),
                    ],
                    dtype=np.float32,
                )
                * emit.random_velocity_scale
            )
            velocities += random_vel
        if self._obs_bounds.size:
            positions, velocities = self._resolve_obstacles_numpy(positions, velocities)
        return positions, velocities

    def _resolve_obstacles_numpy(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve initial overlaps against analytic obstacles (NumPy arrays)."""
        if self._obs_bounds.size == 0:
            return positions, velocities
        x = positions[:, 0]
        y = positions[:, 1]
        u = velocities[:, 0]
        v = velocities[:, 1]
        for i in range(self._obs_bounds.shape[0]):
            xmin, xmax, ymin, ymax = self._obs_bounds[i]
            margin = float(self._obs_margins[i])
            damping = float(self._obs_damping[i])
            inside = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
            if not np.any(inside):
                continue
            xi = np.asarray(x[inside])
            yi = np.asarray(y[inside])
            ui = np.asarray(u[inside])
            vi = np.asarray(v[inside])

            left = xi - xmin
            right = xmax - xi
            bottom = yi - ymin
            top = ymax - yi

            min_lr = np.minimum(left, right)
            min_tb = np.minimum(bottom, top)
            hit_x = min_lr < min_tb
            hit_y = ~hit_x

            if np.any(hit_x):
                left_hit = hit_x & (left < right)
                right_hit = hit_x & ~left_hit
                xi[left_hit] = xmin - margin
                ui[left_hit] *= -damping
                xi[right_hit] = xmax + margin
                ui[right_hit] *= -damping
            if np.any(hit_y):
                bottom_hit = hit_y & (bottom < top)
                top_hit = hit_y & ~bottom_hit
                yi[bottom_hit] = ymin - margin
                vi[bottom_hit] *= -damping
                yi[top_hit] = ymax + margin
                vi[top_hit] *= -damping

            x[inside] = xi
            y[inside] = yi
            u[inside] = ui
            v[inside] = vi
        positions[:, 0] = x
        positions[:, 1] = y
        velocities[:, 0] = u
        velocities[:, 1] = v
        return positions, velocities

    def _create_boundary_particles(self, layers: int = 2) -> np.ndarray:
        xs: list[float] = []
        ys: list[float] = []
        left, bottom = self.domain_min
        right, top = self.domain_max
        for layer in range(layers):
            offset = layer * self.spacing
            x_vals = np.arange(left - offset, right + offset + self.spacing, self.spacing)
            y_vals = np.arange(bottom - offset, top + offset + self.spacing, self.spacing)
            xs.extend(x_vals)
            ys.extend(np.full_like(x_vals, bottom - offset))
            xs.extend(x_vals)
            ys.extend(np.full_like(x_vals, top + offset))
            xs.extend(np.full_like(y_vals, left - offset))
            ys.extend(y_vals)
            xs.extend(np.full_like(y_vals, right + offset))
            ys.extend(y_vals)
        coords = np.column_stack((xs, ys))
        if len(coords) == 0:
            return coords
        unique = np.unique(coords, axis=0)
        return unique.astype(np.float32, copy=False)

    def _setup_solver(
        self,
    ) -> tuple["ParticleArray", "ParticleArray", WCSPHScheme, "Solver", LinkedListNNPS]:
        # Late imports for typing; guarded by constructor check above.
        from pysph.base.utils import get_particle_array_wcsph

        sound_speed = float(self.cfg.sound_speed)
        if sound_speed <= 0:
            raise ValueError("sound_speed must be positive.")
        nu = (
            self.cfg.viscosity / max(self.cfg.rest_density, 1e-6) if self.cfg.viscosity > 0 else 0.0
        )
        configured_sub_dt = self.dt / self.substeps
        # CFL (viscous項含む) でかなり保守的に刻み幅を制限する
        cfl_dt = 0.125 * self.h / sound_speed if sound_speed > 0 else configured_sub_dt
        sub_dt = configured_sub_dt
        if sub_dt > cfl_dt:
            print(
                f"[{self.cfg.name}] sub_dt {sub_dt:.6f} clamped to {cfl_dt:.6f} for CFL (h={self.h:.4f}, c0={sound_speed:.4f})"
            )
            sub_dt = cfl_dt
        steps_per_frame = max(1, int(math.ceil(self.dt / sub_dt)))
        sub_dt = self.dt / steps_per_frame

        fluid = get_particle_array_wcsph(
            name="fluid",
            x=self.positions0[:, 0],
            y=self.positions0[:, 1],
        )
        boundary_coords = np.empty((0, 2), dtype=np.float32)
        solids: list[str] = []
        if self.boundary_mode == BOUNDARY_MODE_WALLS:
            wall_layers = max(3, int(self.cfg.wall_particle_layers))
            boundary_coords = self._create_boundary_particles(layers=wall_layers)
            solids = ["boundary"]
        boundary = get_particle_array_wcsph(
            name="boundary",
            x=boundary_coords[:, 0] if len(boundary_coords) else np.empty(0),
            y=boundary_coords[:, 1] if len(boundary_coords) else np.empty(0),
        )

        for arr in (fluid, boundary):
            arr.m[:] = self.mass
            arr.h[:] = self.h
            arr.rho[:] = self.cfg.rest_density
            arr.rho0[:] = self.cfg.rest_density
            arr.cs[:] = sound_speed

        fluid.u[:] = self.velocities0[:, 0]
        fluid.v[:] = self.velocities0[:, 1]
        boundary.u[:] = 0.0
        boundary.v[:] = 0.0

        scheme = WCSPHScheme(
            fluids=["fluid"],
            solids=solids,
            dim=2,
            rho0=self.cfg.rest_density,
            c0=sound_speed,
            h0=self.h,
            hdx=max(self.cfg.kernel_radius_scale, 1.0),
            gx=float(self.gravity[0]),
            gy=float(self.gravity[1]),
            alpha=0.1,
            beta=0.0,
            nu=nu,
            summation_density=False,
        )
        tf = max(0.0, (self.cfg.timesteps - 1) * self.dt)
        scheme.configure_solver(dt=sub_dt, tf=tf, pfreq=0, n_damp=0)
        print(
            f"[{self.cfg.name}] h {self.h:.4f} dx {self.spacing:.4f} c0 {sound_speed:.4f} sub_dt {sub_dt:.6f} steps/frame {steps_per_frame}"
        )
        solver = scheme.solver
        solver.set_disable_output(True)
        solver.set_output_at_times([])
        solver.set_print_freq(int(1e9))  # effectively silences progress dumps
        solver.set_max_steps(int(max(1, math.ceil(tf / sub_dt) * 2)))
        self._steps_per_frame = steps_per_frame
        self._solver_dt = sub_dt
        if not hasattr(solver, "pm"):
            solver.pm = None  # PySPH < 1.0b2 safety: ensure attribute exists
        if self.domain_manager is not None:
            if hasattr(solver, "set_domain_manager"):
                solver.set_domain_manager(self.domain_manager)
            elif hasattr(solver, "domain_manager"):
                solver.domain_manager = self.domain_manager

        particles = [fluid, boundary]
        scheme.setup_properties(particles, clean=True)
        equations = scheme.get_equations()
        nnps_kwargs: dict[str, Any] = {"dim": 2, "particles": particles, "radius_scale": 2.0}
        if self.domain_manager is not None:
            # Try different parameter names for domain manager (PySPH version compatibility)
            try:
                nnps = LinkedListNNPS(domain_manager=self.domain_manager, **nnps_kwargs)
            except TypeError:
                try:
                    nnps = LinkedListNNPS(domain=self.domain_manager, **nnps_kwargs)
                except TypeError:
                    raise RuntimeError(
                        "LinkedListNNPS does not accept a domain manager; "
                        "periodic boundaries cannot be enabled."
                    )
        else:
            nnps = LinkedListNNPS(**nnps_kwargs)
        solver.setup(particles, equations, nnps)
        solver.add_post_step_callback(self._on_post_step)
        return fluid, boundary, scheme, solver, nnps

    def _build_domain_manager(self) -> Optional["DomainManager"]:
        if self.boundary_mode != BOUNDARY_MODE_PERIODIC:
            return None
        if DomainManager is None:
            raise RuntimeError(
                "Periodic boundary requested but PySPH DomainManager is unavailable."
            )
        if np.any(self.domain_max <= self.domain_min):
            raise ValueError("Invalid domain bounds for periodic boundary.")
        kwargs = {
            "xmin": float(self.domain_min[0]),
            "xmax": float(self.domain_max[0]),
            "ymin": float(self.domain_min[1]),
            "ymax": float(self.domain_max[1]),
            "zmin": 0.0,
            "zmax": 0.0,
            "periodic_in_x": True,
            "periodic_in_y": True,
            "periodic_in_z": False,
        }
        filtered = _filter_kwargs(DomainManager, kwargs)
        if "periodic_in_x" not in filtered or "periodic_in_y" not in filtered:
            raise RuntimeError(
                "PySPH DomainManager does not expose periodic flags; "
                "cannot enable accurate periodic boundaries."
            )
        return DomainManager(**filtered)

    def _apply_tank_motion(self, solver: "Solver") -> None:
        """タンクの水平揺動を外力として速度に加える。"""
        if self.tank_motion is None or self._tank_axis_idx is None:
            return
        dt = float(getattr(solver, "dt", self._solver_dt))
        if dt <= 0:
            return
        accel = float(self.tank_motion.acceleration(float(getattr(solver, "t", 0.0))))
        if accel == 0.0:
            return
        target = self.fluid.u if self._tank_axis_idx == 0 else self.fluid.v
        target[:] = target + accel * dt

    def _record_frame(self) -> None:
        positions = np.stack((np.asarray(self.fluid.x), np.asarray(self.fluid.y)), axis=1).astype(
            np.float32, copy=False
        )
        self.history.append(positions.copy())

    def _on_post_step(self, solver: "Solver") -> None:
        self._substep_counter += 1
        self._apply_tank_motion(solver)
        if self._substep_counter % self._steps_per_frame == 0:
            self._record_frame()
        if len(self.history) >= self._target_frames:
            solver.set_final_time(solver.t)
            solver.set_max_steps(self._substep_counter)

    def rollout(self, timesteps: int) -> np.ndarray:
        self._target_frames = max(1, timesteps)
        self.history = []
        self._substep_counter = 0
        self.nnps.update()
        self._record_frame()
        if timesteps <= 1:
            return np.asarray(self.history, dtype=np.float32)

        tf = max(0.0, (timesteps - 1) * self.dt)
        sub_dt = self.dt / self.substeps
        self.solver.set_final_time(tf)
        self.solver.set_time_step(sub_dt)
        self.solver.set_max_steps(int(max(1, math.ceil(tf / sub_dt) * 2)))

        self.solver.solve(show_progress=False)
        result = np.asarray(self.history, dtype=np.float32)
        if result.shape[0] < timesteps:
            pad = np.repeat(result[-1][None, :, :], timesteps - result.shape[0], axis=0)
            result = np.concatenate([result, pad], axis=0)
        return result[:timesteps]


def generate_scene(
    cfg: FluidCaseConfig, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = random.Random(seed)
    sim = PySPHSimulation(cfg, rng)
    positions = sim.rollout(cfg.timesteps)
    dynamic_count = positions.shape[1]

    wall_coords = np.empty((0, 2), dtype=np.float32)
    wall_layers = max(0, int(cfg.wall_particle_layers))
    if cfg.boundary_mode != BOUNDARY_MODE_PERIODIC:
        wall_layers = max(3, wall_layers)
        if wall_layers > 0:
            wall_coords = sim._create_boundary_particles(layers=wall_layers)
            if len(wall_coords):
                wall_positions = np.repeat(wall_coords[None, :, :], positions.shape[0], axis=0)
                positions = np.concatenate([positions, wall_positions], axis=1)

    n_wall = int(wall_coords.shape[0])
    particle_types = np.zeros(dynamic_count + n_wall, dtype=np.int32)
    if n_wall > 0:
        particle_types[dynamic_count:] = KINEMATIC_PARTICLE_ID

    meta = {
        "seed": seed,
        "dt": cfg.dt,
        "substeps": cfg.solver_substeps,
        "particle_spacing": cfg.particle_spacing,
        "rest_density": cfg.rest_density,
        "stiffness": cfg.stiffness,
        "sound_speed": cfg.sound_speed,
        "viscosity": cfg.viscosity,
        "timesteps": cfg.timesteps,
        "bounds": cfg.domain.as_array().tolist(),
        "gravity": list(cfg.gravity),
        "boundary_damping": cfg.boundary_damping,
        "boundary_augment": cfg.boundary_clamp_limit,
        "wall_particles": int(n_wall),
        "wall_particle_layers": int(wall_layers),
        "boundary_mode": cfg.boundary_mode,
        "note": "synthetic 2D SPH fluid scene",
        "simulator": "pysph_wcsph",
        "sph_backend": "PySPH",
    }
    if cfg.tank_motion is not None:
        meta["tank_motion"] = {
            "axis": cfg.tank_motion.axis,
            "amplitude": float(cfg.tank_motion.amplitude),
            "frequency": float(cfg.tank_motion.frequency),
            "phase": float(cfg.tank_motion.phase),
            "ramp_time": float(cfg.tank_motion.ramp_time),
            "offset": float(cfg.tank_motion.offset),
            "description": "sinusoidal horizontal acceleration applied to fluid for tank sloshing",
        }
    if cfg.obstacles:
        meta["obstacles"] = [
            {
                "center": list(obs.center),
                "size": list(obs.size),
                "padding": float(obs.padding),
                "damping": (
                    float(obs.damping) if obs.damping is not None else float(cfg.boundary_damping)
                ),
                "layers": obs.layers,
            }
            for obs in cfg.obstacles
        ]
    return positions, particle_types, meta


def save_scene_npz(
    out_dir: Path,
    scene_idx: int,
    positions: np.ndarray,
    particle_types: np.ndarray,
    meta: dict[str, Any],
    split: str,
) -> Path:
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"scene_{scene_idx:03d}.npz"
    np.savez_compressed(
        path,
        position=positions.astype(np.float32),
        particle_type=particle_types.astype(np.int32),
        meta=np.array(json.dumps(meta)),
    )
    return path


def _load_scene_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    pos = data["position"]
    ptypes = data["particle_type"]
    meta_raw = data["meta"][()]
    try:
        meta = json.loads(str(meta_raw))
    except Exception:
        meta = {}
    return pos, ptypes, meta


def _generate_split(
    cfg: FluidCaseConfig,
    out_dir: Path,
    split: str,
    num_scenes: int,
    seed_offset: int,
) -> None:
    if num_scenes <= 0:
        return
    print(f"\n[{cfg.name}] Generating {num_scenes} {split} scenes...")
    entries: list[tuple[np.ndarray, np.ndarray, dict[str, Any]] | None] = [
        None for _ in range(num_scenes)
    ]

    split_dir = out_dir / split
    existing_files = sorted(split_dir.glob("scene_*.npz"))
    for path in existing_files:
        try:
            idx = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if idx < 0 or idx >= num_scenes:
            continue
        pos, ptypes, meta = _load_scene_npz(path)
        entries[idx] = (pos, ptypes, meta)

    if existing_files:
        kept = sum(e is not None for e in entries)
        skipped = len(existing_files) - kept
        print(f"[{cfg.name}] Found {kept}/{num_scenes} existing {split} scenes (skipped {skipped})")

    last_meta: dict[str, Any] | None = None
    for idx in range(num_scenes):
        if entries[idx] is not None:
            last_meta = entries[idx][2]
            continue
        scene_seed = cfg.seed + seed_offset + idx
        positions, particle_types, meta = generate_scene(cfg, scene_seed)
        save_scene_npz(out_dir, idx, positions, particle_types, meta, split=split)
        entries[idx] = (positions, particle_types, meta)
        last_meta = meta

    # Flatten to lists (only filled entries)
    trajectories: list[np.ndarray] = []
    particle_types_list: list[np.ndarray] = []
    for entry in entries:
        if entry is None:
            continue
        positions, particle_types, _ = entry
        trajectories.append(positions)
        particle_types_list.append(particle_types)

    if trajectories and last_meta is not None:
        extra_metadata = {
            key: last_meta[key]
            for key in (
                "particle_spacing",
                "rest_density",
                "stiffness",
                "sound_speed",
                "viscosity",
                "bounds",
                "boundary_augment",
                "gravity",
                "wall_particles",
                "wall_particle_layers",
                "boundary_mode",
                "tank_motion",
            )
            if key in last_meta
        }
        dataset_path, meta_path = dataset_utils.export_dataset(
            trajectories,
            particle_types_list,
            out_dir,
            split=split,
            dt=cfg.dt,
            extra_metadata=extra_metadata,
        )
        print(f"[{cfg.name}] Saved {split} dataset to {dataset_path}")
        if split == "train":
            print(f"[{cfg.name}] Wrote metadata to {meta_path}")


def run_generation(
    config_path: Path,
    output_root: Path | None = None,
    target_cases: list[str] | None = None,
    inline_case: str | None = None,
) -> None:
    if inline_case is None:
        dataset_cfg = load_config(config_path)
    else:
        base_raw = _load_yaml(config_path)
        raw_case = _load_inline_case(inline_case)
        assembled_raw = {
            "output_root": base_raw.get("output_root", "./datasets/out_fluid"),
            "defaults": base_raw.get("defaults", {}),
            "cases": [raw_case],
        }
        dataset_cfg = _build_dataset_config(assembled_raw)

    if output_root is not None:
        dataset_cfg.output_root = output_root.resolve()

    cases_to_generate = dataset_cfg.cases
    if inline_case is None and target_cases:
        available_names = {c.name for c in dataset_cfg.cases}
        invalid = [name for name in target_cases if name not in available_names]
        if invalid:
            raise ValueError(
                f"Invalid case name(s): {invalid}. Available cases: {sorted(available_names)}"
            )
        cases_to_generate = [c for c in dataset_cfg.cases if c.name in target_cases]
        print(f"\n=== Generating {len(cases_to_generate)} case(s): {target_cases} ===")

    for case_cfg in cases_to_generate:
        case_dir = case_cfg.resolve_output_dir(dataset_cfg.output_root)
        _ensure_dir(case_dir)
        print(f"\n=== Generating case '{case_cfg.name}' at {case_dir} ===")
        _generate_split(
            case_cfg,
            case_dir,
            split="train",
            num_scenes=case_cfg.num_train_scenes,
            seed_offset=0,
        )
        _generate_split(
            case_cfg,
            case_dir,
            split="valid",
            num_scenes=case_cfg.num_valid_scenes,
            seed_offset=case_cfg.num_train_scenes,
        )
        valid_npz = case_dir / "valid.npz"
        test_npz = case_dir / "test.npz"
        if valid_npz.exists():
            with valid_npz.open("rb") as src, test_npz.open("wb") as dst:
                dst.write(src.read())
            print(f"[{case_cfg.name}] Duplicated {valid_npz.name} -> {test_npz.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fluid datasets using a simple SPH simulation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("datasets/config/fluid.yaml"),
        help="Path to fluid dataset config YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output root directory defined in the config.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Specify case name(s) to generate. If omitted, all cases in config are generated.",
    )
    parser.add_argument(
        "--inline-case",
        type=str,
        default=None,
        help=(
            "YAML/JSON string or file path for a single case. "
            "Merges with defaults in the config file and ignores --cases."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_generation(args.config, args.output_root, args.cases, inline_case=args.inline_case)


if __name__ == "__main__":
    main()
