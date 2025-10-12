# scripts/gen_pymunk.py
import json
import math
import random
from pathlib import Path
import h5py
import numpy as np
import pymunk
from pymunk.vec2d import Vec2d

RNG = random.Random(42)
NP_RNG = np.random.default_rng(42)

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
    else:
        # Segmentなどはスキップ
        pts = np.zeros((0, 2), dtype=np.float32)
    return pts  # (M,2) ローカル


def world_from_local(body, local_pts):
    if len(local_pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    c, s = math.cos(body.angle), math.sin(body.angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (local_pts @ R.T) + np.array(body.position, dtype=np.float32)


def point_velocity(body, world_pts):
    # v_point = v_cm + omega x r  in 2D: omega*z × r(x,y) = (-omega*y, omega*x)
    if len(world_pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    r = world_pts - np.array(body.position, dtype=np.float32)
    omega = body.angular_velocity
    cross = np.stack([-omega * r[:, 1], omega * r[:, 0]], axis=1)
    v = np.array(body.velocity, dtype=np.float32)
    return v + cross


def generate_scene(T=240, dt=1 / 120, substeps=4, n_bodies=(4, 8), density=16):
    space = make_space()
    add_static_bounds(space)

    # 剛体生成
    nb = RNG.randint(*n_bodies)
    bodies, shapes, local_pts = [], [], []
    for _ in range(nb):
        b, s = add_random_rigid(space)
        bodies.append(b)
        shapes.append(s)
        local_pts.append(sample_points_on_shape(s, density=density))
    # サンプル点を連結して粒子化
    body_offsets = []
    offset = 0
    for pts in local_pts:
        body_offsets.append((offset, offset + len(pts)))
        offset += len(pts)
    N = offset

    positions = np.zeros((T, N, 2), np.float32)
    velocities = np.zeros((T, N, 2), np.float32)
    node_attr = np.zeros(
        (N, 6), np.float32
    )  # [mass, type_onehot(2), body_id, elasticity, friction]
    globals_arr = np.zeros((T, 3), np.float32)  # [gx, gy, dt]

    # 固定dtで進めつつ記録
    sub_dt = dt / substeps
    for t in range(T):
        for _ in range(substeps):
            space.step(sub_dt)

        # 記録
        idx = 0
        for i, (b, s) in enumerate(zip(bodies, shapes)):
            pts_local = local_pts[i]
            pts_world = world_from_local(b, pts_local)
            positions[t, idx : idx + len(pts_local)] = pts_world
            velocities[t, idx : idx + len(pts_local)] = point_velocity(b, pts_world)

            # 固定属性（最初のフレームだけ埋める）
            if t == 0:
                mass = b.mass
                body_id = float(i)
                elast = getattr(s, "elasticity", 0.0)
                fric = getattr(s, "friction", 0.5)
                # type_onehot: [rigid, static]
                node_attr[idx : idx + len(pts_local), 0] = mass
                node_attr[idx : idx + len(pts_local), 1] = 1.0  # rigid=1
                node_attr[idx : idx + len(pts_local), 2] = 0.0  # static=0
                node_attr[idx : idx + len(pts_local), 3] = body_id
                node_attr[idx : idx + len(pts_local), 4] = elast
                node_attr[idx : idx + len(pts_local), 5] = fric

            idx += len(pts_local)

        globals_arr[t] = np.array([space.gravity[0], space.gravity[1], dt], np.float32)

    # 教師（速度教師にする例：v_{t+1}を正解に）
    target = np.zeros_like(velocities)
    target[:-1] = velocities[1:]
    target[-1] = velocities[-1]  # 末尾はコピーでOK（使用時に除外でも可）

    meta = dict(
        seed=42,
        nb=len(bodies),
        dt=dt,
        substeps=substeps,
        note="pymunk random drop scene",
    )
    return positions, velocities, node_attr, globals_arr, target, meta


def save_h5(
    path,
    scene_idx,
    positions,
    velocities,
    node_attr,
    globals_arr,
    target,
    meta,
    split="train",
):
    with h5py.File(path, "a") as f:
        grp = f.require_group(f"/{split}/scene_{scene_idx:03d}")
        grp.create_dataset("positions", data=positions, compression="gzip")
        grp.create_dataset("velocities", data=velocities, compression="gzip")
        grp.create_dataset("node_attr", data=node_attr, compression="gzip")
        grp.create_dataset("globals", data=globals_arr, compression="gzip")
        grp.create_dataset("target", data=target, compression="gzip")
        grp.attrs["meta"] = json.dumps(meta)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "out" / "train.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    for i in range(50):  # 50シーンだけ試しに
        pos, vel, na, glb, tgt, meta = generate_scene()
        save_h5(out, i, pos, vel, na, glb, tgt, meta, split="train")
    print("saved:", out)
