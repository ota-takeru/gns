import json
import math
import random
from pathlib import Path
import numpy as np
import pymunk

RNG = random.Random(42)


def make_space(g=(0.0, -9.81)):
    space = pymunk.Space()
    space.gravity = g
    space.iterations = 30  # 収束反復。接触安定性に効く
    return space


def add_static_bounds(space, left=-5, right=5, bottom=0, top=8):
    static_body = space.static_body
    segs = [
        pymunk.Segment(static_body, (left, bottom), (right, bottom), 0.0),  # floor
        pymunk.Segment(static_body, (left, bottom), (left, top), 0.0),
        pymunk.Segment(static_body, (right, bottom), (right, top), 0.0),
    ]
    for s in segs:
        s.friction = 0.6
        s.elasticity = 0.0
    space.add(*segs)
    return segs


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


def generate_scene(T=240, dt=1 / 120, substeps=4, n_bodies=(4, 8), density=16):
    space = make_space()
    static_shapes = add_static_bounds(space)

    # 剛体生成
    nb = RNG.randint(*n_bodies)
    bodies, local_pts = [], []
    for _ in range(nb):
        b, s = add_random_rigid(space)
        bodies.append(b)
        local_pts.append(sample_points_on_shape(s, density=density))
    # 境界（静的剛体）も粒子化
    static_body = space.static_body
    static_pts = [
        sample_points_on_shape(s, density=density)
        for s in static_shapes
        if s is not None
    ]
    static_pts = [pts for pts in static_pts if len(pts) > 0]
    static_pts_all = (
        np.concatenate(static_pts, axis=0)
        if len(static_pts) > 0
        else np.zeros((0, 2), dtype=np.float32)
    )

    entries = []
    N = 0
    for rigid_idx, (body, pts_local) in enumerate(zip(bodies, local_pts)):
        if len(pts_local) == 0:
            continue
        entries.append((body, pts_local, rigid_idx))
        N += len(pts_local)

    if len(static_pts_all) > 0:
        static_rigid_id = nb
        entries.append((static_body, static_pts_all, static_rigid_id))
        N += len(static_pts_all)

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

    meta = dict(
        seed=42,
        nb=len(bodies),
        dt=dt,
        substeps=substeps,
        note="pymunk random drop scene",
        static_particles=int(len(static_pts_all)),
        dynamic_particles=int(N - len(static_pts_all)),
    )
    return positions, rigid_ids, meta


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


if __name__ == "__main__":
    out_root = Path(__file__).resolve().parents[1] / "out"
    num_scenes = 5 # シーン数
    split = "train"
    for i in range(num_scenes):
        pos, rigid_ids, meta = generate_scene()
        save_npz(out_root, i, pos, rigid_ids, meta, split=split)
    print(f"Generated {num_scenes} scenes in {out_root / split}")
