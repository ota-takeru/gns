import numpy as np
import matplotlib.pyplot as plt

def plot_time_series(pos, vel, idx_n=0, save=None, title="pos/vel over time"):
    """1粒子のx(t), v_x(t)の波形。まずはこれで崩れを検知。"""
    T = pos.shape[0]
    t = np.arange(T)
    fig, ax = plt.subplots(2, 1, figsize=(6,4), sharex=True)
    ax[0].plot(t, pos[:, idx_n, 0]); ax[0].set_ylabel("x")
    ax[1].plot(t, vel[:, idx_n, 0]); ax[1].set_ylabel("vx")
    ax[1].set_xlabel("t (step)")
    fig.suptitle(title); fig.tight_layout()
    if save: fig.savefig(save, dpi=150)
    return fig

