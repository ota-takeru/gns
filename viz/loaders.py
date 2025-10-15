import json
import numpy as np


def load_episode(npz_path, *, include_rigid_id=False):
    with np.load(npz_path, allow_pickle=True) as d:
        if "position" in d:
            pos = np.array(d["position"], copy=False)
        elif "pos" in d:
            pos = np.array(d["pos"], copy=False)
        else:
            raise KeyError("npz file must contain 'position' or 'pos'")

        if "velocity" in d:
            vel = np.array(d["velocity"], copy=False)
        elif "vel" in d:
            vel = np.array(d["vel"], copy=False)
        else:
            vel = None

        rigid_id = np.array(d["rigid_id"], copy=False) if "rigid_id" in d else None

        if "dt" in d:
            dt_arr = d["dt"]
            dt = float(dt_arr[0]) if np.ndim(dt_arr) > 0 else float(np.array(dt_arr).item())
        else:
            meta = None
            if "meta" in d:
                meta_raw = d["meta"]
                if isinstance(meta_raw, np.ndarray):
                    meta_raw = meta_raw.item()
                try:
                    meta = json.loads(meta_raw)
                except Exception:
                    meta = None
            dt = float(meta.get("dt")) if isinstance(meta, dict) and "dt" in meta else 0.02

        if "g" in d:
            g = np.array(d["g"], dtype=np.float32, copy=False)
        else:
            g = np.array([0.0, -9.8], dtype=np.float32)

    if include_rigid_id:
        return pos, vel, dt, g, rigid_id
    return pos, vel, dt, g
