import numpy as np

def load_episode(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    pos = d["pos"]   # [T, N, 2]
    vel = d["vel"]   # [T, N, 2]
    dt  = float(d["dt"][0]) if "dt" in d else 0.02
    g   = d["g"] if "g" in d else np.array([0.0, -9.8], dtype=np.float32)
    return pos, vel, dt, g

