import numpy as np

def gen_freefall(T=50, dt=0.02, g=-9.8, x0=0.0, v0=5.0):
    xs, vs = [x0], [v0]
    for _ in range(T-1):
        v = vs[-1] + dt*g
        x = xs[-1] + dt*v
        xs.append(x); vs.append(v)
    # 2D拡張に備えてshapeを揃える: pos=[T,1,2], vel=[T,1,2]
    pos = np.zeros((T,1,2), dtype=np.float32); pos[...,0]=np.array(xs)[:,None]
    vel = np.zeros((T,1,2), dtype=np.float32); vel[...,0]=np.array(vs)[:,None]
    globals = {"dt": dt, "g": [0.0, g]}
    return pos, vel, globals

# if __name__ == "__main__":
#     pos, vel, gl = gen_freefall()
#     np.savez("freefall.npz", pos=pos, vel=vel, g=np.array(gl["g"]), dt=np.array([gl["dt"]]))
