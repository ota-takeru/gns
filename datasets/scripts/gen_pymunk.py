import json
import math
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pymunk

RNG = random.Random(42)


def make_space(g=(0.0, -9.81)):
    space = pymunk.Space()
    space.gravity = g
    space.iterations = 30  # 収束反復。接触安定性に効く
    return space


def add_wall_as_rigid(space, left=-5, right=5, bottom=0, top=8):
    """壁を剛体として追加（粒子表現のため）- STATICボディとして固定"""
    walls = []

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

    return walls


def add_random_rigid(space, x_range=(-3, 3), y=5.0):
    kind = RNG.choice(["box", "circle"])
    mass = RNG.uniform(0.5, 3.0)
    if kind == "box":
        w, h = RNG.uniform(0.3, 0.8), RNG.uniform(0.3, 0.8)
        moment = pymunk.moment_for_box(mass, (w, h))
        body = pymunk.Body(mass, moment)
        body.position = (RNG.uniform(*x_range), y + RNG.uniform(0, 2))
        body.angle = RNG.uniform(-math.pi, math.pi)
        shape = pymunk.Poly.create_box(body, (w, h))
    else:
        r = RNG.uniform(0.2, 0.5)
        moment = pymunk.moment_for_circle(mass, 0, r)
        body = pymunk.Body(mass, moment)
        body.position = (RNG.uniform(*x_range), y + RNG.uniform(0, 2))
        shape = pymunk.Circle(body, r)

    shape.friction = RNG.uniform(0.2, 0.9)
    shape.elasticity = RNG.uniform(0.0, 0.4)
    space.add(body, shape)
    return body, shape


def sample_points_on_shape(shape, density=12):
    """剛体shapeの境界/内部にサンプル点（ローカル座標）を作る"""
    if isinstance(shape, pymunk.Circle):
        r = shape.radius
        angles = np.linspace(0, 2 * np.pi, num=density, endpoint=False)
        pts = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
    elif isinstance(shape, pymunk.Poly):
        verts = np.array(shape.get_vertices(), dtype=np.float32)  # ローカル
        # 辺上を等間隔サンプリング（簡易）
        segs = []
        for i in range(len(verts)):
            a = verts[i]
            b = verts[(i + 1) % len(verts)]
            for t in np.linspace(
                0, 1, num=max(2, density // len(verts)), endpoint=False
            ):
                segs.append(a * (1 - t) + b * t)
        pts = np.array(segs, dtype=np.float32)
    elif isinstance(shape, pymunk.Segment):
        a = np.array(shape.a, dtype=np.float32)
        b = np.array(shape.b, dtype=np.float32)
        num = max(2, density)
        ts = np.linspace(0.0, 1.0, num=num, endpoint=False)
        pts = (1.0 - ts)[:, None] * a + ts[:, None] * b
    else:
        # その他のshapeタイプには未対応
        pts = np.zeros((0, 2), dtype=np.float32)
    return pts  # (M,2) ローカル


def world_from_local(body, local_pts):
    if len(local_pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    c, s = math.cos(body.angle), math.sin(body.angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (local_pts @ R.T) + np.array(body.position, dtype=np.float32)


def generate_scene(T=240, dt=1 / 120, substeps=4, n_bodies=(2, 5), density=16):
    space = make_space()
    walls = add_wall_as_rigid(space)

    # 剛体生成（落下する物体）
    nb = RNG.randint(*n_bodies)
    bodies, local_pts = [], []
    for _ in range(nb):
        b, s = add_random_rigid(space)
        bodies.append(b)
        local_pts.append(sample_points_on_shape(s, density=density))

    # 壁も剛体として扱う
    wall_bodies = []
    wall_local_pts = []
    for wall_body, wall_shape in walls:
        wall_bodies.append(wall_body)
        wall_local_pts.append(sample_points_on_shape(wall_shape, density=density))

    # 全剛体をentriesに追加（落下する物体 + 壁）
    entries = []
    N = 0

    # 落下する剛体
    for rigid_idx, (body, pts_local) in enumerate(zip(bodies, local_pts)):
        if len(pts_local) == 0:
            continue
        entries.append((body, pts_local, rigid_idx))
        N += len(pts_local)

    # 壁の剛体
    for wall_idx, (wall_body, pts_local) in enumerate(zip(wall_bodies, wall_local_pts)):
        if len(pts_local) == 0:
            continue
        wall_rigid_id = nb + wall_idx
        entries.append((wall_body, pts_local, wall_rigid_id))
        N += len(pts_local)

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

    # 壁の粒子数を計算
    wall_particles = sum(len(pts) for pts in wall_local_pts)
    dynamic_particles = N - wall_particles

    meta = dict(
        seed=42,
        nb=len(bodies),  # 落下する剛体の数
        n_walls=len(walls),  # 壁の数
        total_rigids=len(bodies) + len(walls),  # 総剛体数
        dt=dt,
        substeps=substeps,
        note="pymunk random drop scene with walls as rigids",
        wall_particles=int(wall_particles),
        dynamic_particles=int(dynamic_particles),
    )
    return positions, rigid_ids, meta


def _extract_all_positions_with_types(positions, rigid_ids, n_dynamic_rigid):
    """Extract all positions and assign particle types (0=dynamic, 3=kinematic/wall)."""
    # Dynamic particles: particle_type = 0
    # Wall particles: particle_type = 3 (KINEMATIC_PARTICLE_ID)
    particle_types = np.where(rigid_ids < n_dynamic_rigid, 0, 3).astype(np.int32)
    return positions.astype(np.float32, copy=False), particle_types


def _compute_differences(sequence):
    """Compute first-order differences along the time axis without dt scaling."""
    return sequence[1:] - sequence[:-1]


def _collect_statistics(trajectories):
    """Aggregate velocity/acceleration stats across trajectories for metadata."""
    velocity_rows = []
    acceleration_rows = []
    for pos in trajectories:
        vel = _compute_differences(pos)
        if vel.size:
            velocity_rows.append(vel.reshape(-1, vel.shape[-1]))
            acc = _compute_differences(vel)
            if acc.size:
                acceleration_rows.append(acc.reshape(-1, acc.shape[-1]))

    if velocity_rows:
        vel_stack = np.concatenate(velocity_rows, axis=0)
        vel_mean = vel_stack.mean(axis=0)
        vel_std = vel_stack.std(axis=0)
    else:
        vel_mean = vel_std = np.zeros((trajectories[0].shape[-1],), dtype=np.float32)

    if acceleration_rows:
        acc_stack = np.concatenate(acceleration_rows, axis=0)
        acc_mean = acc_stack.mean(axis=0)
        acc_std = acc_stack.std(axis=0)
    else:
        acc_mean = acc_std = np.zeros((trajectories[0].shape[-1],), dtype=np.float32)

    return vel_mean, vel_std, acc_mean, acc_std


def _estimate_connectivity_radius(trajectories):
    """Heuristic connectivity radius based on nearest-neighbour distance."""
    nearest_distances = []
    for pos in trajectories:
        frames = pos[: min(5, pos.shape[0])]
        for frame in frames:
            n = frame.shape[0]
            if n < 2:
                continue
            diff = frame[:, None, :] - frame[None, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            dist[np.eye(n, dtype=bool)] = np.inf  # ignore self-distance
            nn = np.min(dist, axis=1)
            nn = nn[np.isfinite(nn)]
            if nn.size:
                nearest_distances.append(nn)

    if not nearest_distances:
        return 1.0

    all_nn = np.concatenate(nearest_distances, axis=0)
    # Use a generous multiplier so that the message graph connects over shapes.
    return float(np.percentile(all_nn, 90) * 1.5)


def export_dataset(trajectories, particle_types_list, out_dir, split, dt):
    """Save aggregated dataset compatible with data_loader along with metadata.

    Note: We store particle_types as an array per trajectory to support mixed types
    (dynamic and kinematic particles in the same scene).
    """
    if not trajectories:
        raise ValueError("No trajectories provided for dataset export.")

    entries = []
    for pos, ptypes in zip(trajectories, particle_types_list):
        # Store particle type array instead of a single value
        # This allows mixing dynamic (0) and kinematic (3) particles
        entries.append((pos.astype(np.float32, copy=False), ptypes.astype(np.int32)))

    gns_data = np.array(entries, dtype=object)
    dataset_path = out_dir / f"{split}.npz"
    np.savez_compressed(dataset_path, gns_data=gns_data)

    vel_mean, vel_std, acc_mean, acc_std = _collect_statistics(trajectories)
    dim = int(trajectories[0].shape[-1])
    input_sequence_length = 6  # Must match INPUT_SEQUENCE_LENGTH in training.
    particle_type_embedding = (
        16  # Must match particle_type_embedding_size in simulator.
    )
    velocity_feature_dim = dim * (input_sequence_length - 1)
    metadata = {
        "dim": int(trajectories[0].shape[-1]),
        "sequence_length": int(trajectories[0].shape[0]),
        "dt": float(dt),
        "default_connectivity_radius": _estimate_connectivity_radius(trajectories),
        "vel_mean": vel_mean.tolist(),
        "vel_std": vel_std.tolist(),
        "acc_mean": acc_mean.tolist(),
        "acc_std": acc_std.tolist(),
        "nnode_in": int(velocity_feature_dim + particle_type_embedding),
        "nedge_in": int(dim + 1),
    }
    meta_path = out_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump({"train": metadata, "rollout": metadata}, fp, indent=2)

    return dataset_path, meta_path


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


def main(out_dir: Optional[Union[Path, str]] = None):
    out_dir = Path(out_dir) if out_dir is not None else Path("out")

    # Train データセットの生成
    num_train_scenes = 5000  # 訓練シーン数（テスト用に少なく）
    split = "train"
    trajectories = []
    particle_types_list = []
    last_meta = None
    print(f"Generating {num_train_scenes} training scenes...")
    for i in range(num_train_scenes):
        pos, rigid_ids, meta = generate_scene()
        save_npz(out_root, i, pos, rigid_ids, meta, split=split)
        # 全粒子を保持し、粒子タイプを割り当て
        all_pos, particle_types = _extract_all_positions_with_types(
            pos, rigid_ids, meta["nb"]
        )
        trajectories.append(all_pos)
        particle_types_list.append(particle_types)
        last_meta = meta

    if trajectories and last_meta is not None:
        dataset_path, meta_path = export_dataset(
            trajectories, particle_types_list, out_root, split=split, dt=last_meta["dt"]
        )
        print(f"Saved train dataset to {dataset_path}")
        print(f"Wrote metadata to {meta_path}")
    print(f"Generated {num_train_scenes} scenes in {out_root / split}")

    # Valid データセットの生成
    num_valid_scenes = 500  # 検証シーン数
    split = "valid"
    trajectories = []
    particle_types_list = []
    last_meta = None
    print(f"\nGenerating {num_valid_scenes} validation scenes...")
    for i in range(num_valid_scenes):
        pos, rigid_ids, meta = generate_scene()
        save_npz(out_root, i, pos, rigid_ids, meta, split=split)
        # 全粒子を保持し、粒子タイプを割り当て
        all_pos, particle_types = _extract_all_positions_with_types(
            pos, rigid_ids, meta["nb"]
        )
        trajectories.append(all_pos)
        particle_types_list.append(particle_types)
        last_meta = meta

    if trajectories and last_meta is not None:
        dataset_path, _ = export_dataset(
            trajectories, particle_types_list, out_root, split=split, dt=last_meta["dt"]
        )
        print(f"Saved valid dataset to {dataset_path}")
    print(f"Generated {num_valid_scenes} scenes in {out_root / split}")


if __name__ == "__main__":
    main()
