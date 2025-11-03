import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pymunk
import yaml

try:
    from datasets.scripts import dataset_utils
except ImportError:  # pragma: no cover
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = CURRENT_DIR.parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from datasets.scripts import dataset_utils


@dataclass
class DatasetConfig:
    output_dir: str = "out"
    num_train_scenes: int = 1000
    num_valid_scenes: int = 100
    timesteps: int = 240
    dt: float = 1 / 120
    substeps: int = 4
    rigid_count_range: tuple[int, int] = field(default_factory=lambda: (2, 5))
    particle_density: float = 150.0  # 粒子密度（単位面積当たりの粒子数）
    wall_density_scale: float = 1.0  # 壁の密度スケール（1.0で同じ密度）
    gravity: tuple[float, float] = field(default_factory=lambda: (0.0, -9.81))
    visualization_marker_scale: float = 1.0  # 可視化時のマーカーサイズ倍率
    boundary_clamp_limit: float = 1.0  # 壁距離特徴のクリップ上限
    seed: int = 42


def _load_raw_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def load_dataset_config(
    config_path: Optional[Union[str, Path]] = None,
) -> tuple[DatasetConfig, Optional[Path]]:
    default_path = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg_path = Path(config_path) if config_path is not None else default_path
    raw = _load_raw_config(cfg_path) if cfg_path.exists() else {}

    if "rigid_count_range" not in raw:
        # 後方互換用: n_bodies: [min, max] を許容
        n_bodies = raw.pop("n_bodies", None)
        if isinstance(n_bodies, (list, tuple)) and len(n_bodies) == 2:
            raw["rigid_count_range"] = tuple(int(v) for v in n_bodies)

    cfg = DatasetConfig(**raw)
    return cfg, (cfg_path if cfg_path.exists() else None)


def make_space(g=(0.0, -9.81)):
    space = pymunk.Space()
    space.gravity = g
    space.iterations = 30  # 収束反復。接触安定性に効く
    return space


def add_wall_as_rigid(space, left=-5, right=5, bottom=0, top=8):
    """壁を剛体として追加（物理計算用）。距離計算向けに境界情報も返す。"""
    walls = []
    bounds = np.array([[left, right], [bottom, top]], dtype=np.float32)

    # 床 - STATIC bodyとして作成（完全に固定）
    floor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    floor_body.position = (0, 0)
    floor_shape = pymunk.Segment(floor_body, (left, bottom), (right, bottom), 0.0)
    floor_shape.friction = 0.6
    floor_shape.elasticity = 0.0
    space.add(floor_body, floor_shape)
    walls.append((floor_body, floor_shape))

    # 左壁 - STATIC bodyとして作成（完全に固定）
    left_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    left_body.position = (0, 0)
    left_shape = pymunk.Segment(left_body, (left, bottom), (left, top), 0.0)
    left_shape.friction = 0.6
    left_shape.elasticity = 0.0
    space.add(left_body, left_shape)
    walls.append((left_body, left_shape))

    # 右壁 - STATIC bodyとして作成（完全に固定）
    right_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    right_body.position = (0, 0)
    right_shape = pymunk.Segment(right_body, (right, bottom), (right, top), 0.0)
    right_shape.friction = 0.6
    right_shape.elasticity = 0.0
    space.add(right_body, right_shape)
    walls.append((right_body, right_shape))

    return walls, bounds


def add_random_rigid(space, rng: random.Random, x_range=(-3, 3), y=5.0):
    kind = rng.choice(["box", "circle"])
    mass = rng.uniform(0.5, 3.0)
    if kind == "box":
        w, h = rng.uniform(0.3, 0.8), rng.uniform(0.3, 0.8)
        moment = pymunk.moment_for_box(mass, (w, h))
        body = pymunk.Body(mass, moment)
        body.position = (rng.uniform(*x_range), y + rng.uniform(0, 2))
        body.angle = rng.uniform(-math.pi, math.pi)
        shape = pymunk.Poly.create_box(body, (w, h))
    else:
        r = rng.uniform(0.2, 0.5)
        moment = pymunk.moment_for_circle(mass, 0, r)
        body = pymunk.Body(mass, moment)
        body.position = (rng.uniform(*x_range), y + rng.uniform(0, 2))
        shape = pymunk.Circle(body, r)

    shape.friction = rng.uniform(0.2, 0.9)
    shape.elasticity = rng.uniform(0.0, 0.4)
    space.add(body, shape)
    return body, shape


def _axis_range(min_v: float, max_v: float, spacing: float) -> np.ndarray:
    """Generate coordinates ensuring at least one sample across the interval."""
    if max_v - min_v <= spacing:
        center = (min_v + max_v) * 0.5
        return np.array([center], dtype=np.float32)
    count = max(1, int(math.ceil((max_v - min_v) / spacing)))
    return np.linspace(min_v, max_v, num=count + 1, dtype=np.float32)


def _point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray, tol: float) -> bool:
    seg = end - start
    seg_len = np.linalg.norm(seg)
    if seg_len == 0:
        return np.linalg.norm(point - start) <= tol
    proj = np.dot(point - start, seg) / seg_len
    if proj < -tol or proj > seg_len + tol:
        return False
    closest = start + np.clip(proj, 0.0, seg_len) * seg / seg_len
    return np.linalg.norm(point - closest) <= tol


def _point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    x, y = point
    inside = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def sample_shape_points(
    shape: pymunk.Shape, spacing: float, include_interior: bool = True
) -> np.ndarray:
    """剛体shapeの粒子サンプル（ローカル座標）。"""
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    tol = max(1e-3, spacing * 0.6)

    if isinstance(shape, pymunk.Circle):
        r = shape.radius
        xs = _axis_range(-r, r, spacing)
        ys = _axis_range(-r, r, spacing)
        pts = []
        r_sq = r * r
        for x in xs:
            for y in ys:
                dist_sq = x * x + y * y
                if include_interior and dist_sq <= r_sq + 1e-6:
                    pts.append((x, y))
                elif not include_interior and abs(math.sqrt(dist_sq) - r) <= tol:
                    pts.append((x, y))
        if not pts:
            pts = [(0.0, 0.0)]
        return np.asarray(pts, dtype=np.float32)

    if isinstance(shape, pymunk.Poly):
        verts = np.array(shape.get_vertices(), dtype=np.float32)
        xs = _axis_range(float(verts[:, 0].min()), float(verts[:, 0].max()), spacing)
        ys = _axis_range(float(verts[:, 1].min()), float(verts[:, 1].max()), spacing)
        pts = []
        for x in xs:
            for y in ys:
                p = np.array([x, y], dtype=np.float32)
                if include_interior:
                    if _point_in_polygon(p, verts):
                        pts.append(p)
                        continue
                # check boundary proximity
                for i in range(len(verts)):
                    a = verts[i]
                    b = verts[(i + 1) % len(verts)]
                    if _point_on_segment(p, a, b, tol):
                        pts.append(p)
                        break
        if not pts:
            centroid = verts.mean(axis=0)
            pts = [centroid.astype(np.float32)]
        return np.asarray(pts, dtype=np.float32)

    if isinstance(shape, pymunk.Segment):
        a = np.array(shape.a, dtype=np.float32)
        b = np.array(shape.b, dtype=np.float32)
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len == 0.0:
            return np.array([a], dtype=np.float32)
        num = max(2, int(math.ceil(seg_len / spacing)) + 1)
        ts = np.linspace(0.0, 1.0, num=num, dtype=np.float32)
        pts = (1.0 - ts)[:, None] * a + ts[:, None] * b
        return pts.astype(np.float32)

    # その他のshapeタイプには未対応
    return np.zeros((0, 2), dtype=np.float32)


def world_from_local(body, local_pts):
    if len(local_pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    c, s = math.cos(body.angle), math.sin(body.angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (local_pts @ R.T) + np.array(body.position, dtype=np.float32)


def _density_to_spacing(particle_density: float) -> float:
    if particle_density <= 0:
        raise ValueError("particle_density must be greater than 0")
    return 1.0 / math.sqrt(particle_density)


def generate_scene(cfg: DatasetConfig, rng: random.Random, scene_seed: int):
    space = make_space(cfg.gravity)
    walls, bounds = add_wall_as_rigid(space)

    # 剛体生成（落下する物体）
    rigid_min, rigid_max = cfg.rigid_count_range
    rigid_min, rigid_max = int(rigid_min), int(rigid_max)
    if rigid_min > rigid_max:
        rigid_min, rigid_max = rigid_max, rigid_min
    nb = rng.randint(rigid_min, rigid_max)
    bodies, local_pts = [], []
    particle_spacing = _density_to_spacing(cfg.particle_density)
    for _ in range(nb):
        b, s = add_random_rigid(space, rng=rng)
        bodies.append(b)
        local_pts.append(sample_shape_points(s, spacing=particle_spacing, include_interior=True))

    # 全剛体をentriesに追加（落下する物体のみ）
    entries = []
    N = 0  # 合計粒子数（壁は含めない）
    for rigid_idx, (body, pts_local) in enumerate(zip(bodies, local_pts)):
        if len(pts_local) == 0:
            continue
        entries.append((body, pts_local, rigid_idx))
        N += len(pts_local)

    T = cfg.timesteps
    dt = cfg.dt
    substeps = cfg.substeps

    positions = np.zeros((T, N, 2), np.float32)
    rigid_ids = np.zeros((N,), np.int32)

    # 固定dtで進めつつ記録
    sub_dt = dt / substeps
    for t in range(T):
        for _ in range(substeps):
            space.step(sub_dt)

        # 記録
        idx = 0
        for body, pts_local, rigid_id in entries:
            pts_world = world_from_local(body, pts_local)
            positions[t, idx : idx + len(pts_local)] = pts_world

            if t == 0:
                rigid_ids[idx : idx + len(pts_local)] = rigid_id

            idx += len(pts_local)

    dynamic_particles = N

    meta = dict(
        seed=scene_seed,
        nb=len(bodies),  # 落下する剛体の数
        n_walls=len(walls),  # 壁の数
        total_rigids=len(bodies) + len(walls),  # 総剛体数
        dt=dt,
        substeps=substeps,
        note="pymunk random drop scene with static wall boundaries",
        wall_particles=0,
        dynamic_particles=int(dynamic_particles),
        particle_density=float(cfg.particle_density),
        particle_spacing=float(particle_spacing),
        wall_density_scale=float(cfg.wall_density_scale),
        visualization_marker_scale=float(cfg.visualization_marker_scale),
        timesteps=int(T),
        bounds=bounds.tolist(),
        boundary_augment=float(cfg.boundary_clamp_limit),
    )
    return positions, rigid_ids, meta


def _extract_all_positions_with_types(positions, rigid_ids, n_dynamic_rigid):
    """Extract all positions and assign particle types (0=dynamic)."""
    # Wall interactions are modelled via distance-to-boundary features, but we keep the
    # kinematic type ID reserved for backward compatibility with downstream code.
    particle_types = np.where(rigid_ids < n_dynamic_rigid, 0, 3).astype(np.int32)
    return positions.astype(np.float32, copy=False), particle_types


def save_npz(out_dir, scene_idx, positions, rigid_ids, meta, split="train"):
    split_dir = out_dir / split if split else out_dir
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"scene_{scene_idx:03d}.npz"
    np.savez_compressed(
        path,
        position=positions,
        rigid_id=rigid_ids,
        meta=np.array(json.dumps(meta)),
    )
    return path


def main(
    config_path: Optional[Union[Path, str]] = None,
    output_dir: Optional[Union[Path, str]] = None,
):
    cfg, resolved_path = load_dataset_config(config_path)
    if output_dir is not None:
        cfg.output_dir = str(output_dir)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spacing = _density_to_spacing(cfg.particle_density)
    config_msg = f"{resolved_path}" if resolved_path else "defaults"
    print(f"Using dataset config: {config_msg}")
    print(
        f"Particle density={cfg.particle_density:.3f} -> spacing≈{spacing:.4f}, "
        f"wall_scale={cfg.wall_density_scale:.3f}"
    )

    def _generate_split(split: str, num_scenes: int, seed_offset: int):
        print(f"\nGenerating {num_scenes} {split} scenes...")
        trajectories = []
        particle_types_list = []
        last_meta = None
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_scenes):
            scene_seed = cfg.seed + seed_offset + i
            rng = random.Random(scene_seed)
            pos, rigid_ids, meta = generate_scene(cfg, rng, scene_seed)
            save_npz(out_dir, i, pos, rigid_ids, meta, split=split)
            all_pos, particle_types = _extract_all_positions_with_types(
                pos, rigid_ids, meta["nb"]
            )
            trajectories.append(all_pos)
            particle_types_list.append(particle_types)
            last_meta = meta

        if trajectories and last_meta is not None:
            extra_metadata = {
                key: last_meta[key]
                for key in (
                    "particle_spacing",
                    "particle_density",
                    "wall_density_scale",
                    "visualization_marker_scale",
                    "bounds",
                    "boundary_augment",
                )
                if key in last_meta
            }
            dataset_path, meta_path = dataset_utils.export_dataset(
                trajectories,
                particle_types_list,
                out_dir,
                split=split,
                dt=last_meta["dt"],
                extra_metadata=extra_metadata,
            )
            print(f"Saved {split} dataset to {dataset_path}")
            if split == "train":
                print(f"Wrote metadata to {meta_path}")
        print(f"Generated {num_scenes} scenes in {split_dir}")

    _generate_split("train", cfg.num_train_scenes, seed_offset=0)
    _generate_split(
        "valid", cfg.num_valid_scenes, seed_offset=cfg.num_train_scenes
    )

    # Rollout 用に valid の複製を test.npz として用意
    valid_npz = out_dir / "valid.npz"
    test_npz = out_dir / "test.npz"
    if valid_npz.exists():
        shutil.copy2(valid_npz, test_npz)
        print(f"Duplicated {valid_npz} -> {test_npz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset using pymunk scenes.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to dataset config YAML (defaults to datasets/config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory defined in the config.",
    )
    args = parser.parse_args()
    main(config_path=args.config, output_dir=args.output_dir)
