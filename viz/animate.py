import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from pathlib import Path


def animate_points(pos, radius=None, save_mp4="rollout.mp4", fps=30, lim=None, rigid_id=None):
    """
    粒子散布を動画化（教師 or 予測どちらでも）。radius=[N]でサイズ指定可。
    """
    T, N, _ = pos.shape
    fig, ax = plt.subplots(figsize=(5, 5))
    size = 20 if radius is None else (radius * 100) ** 2
    colors = rigid_id if rigid_id is not None and len(np.unique(rigid_id)) > 1 else None
    if colors is None:
        scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=size)
    else:
        scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=size, c=colors, cmap="tab20")
        scat.set_array(colors)
    ax.set_aspect("equal")
    if lim is None:
        xmn, xmx = pos[..., 0].min(), pos[..., 0].max()
        ymn, ymx = pos[..., 1].min(), pos[..., 1].max()
        pad = 0.1 * max(xmx - xmn, ymx - ymn + 1e-6)
        lim = (xmn - pad, xmx + pad, ymn - pad, ymx + pad)
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.grid(True, ls=":")

    def update(t):
        scat.set_offsets(pos[t])
        return (scat,)

    ani = FuncAnimation(fig, update, frames=T, blit=True)
    out_path = Path(save_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = FFMpegWriter(fps=fps)
        ani.save(out_path, writer=writer, dpi=150)
    except Exception:
        # ffmpeg無い環境ではHTML5アニメを返す
        from matplotlib.animation import HTMLWriter

        html_path = out_path.with_suffix(".html")
        ani.save(html_path, writer=HTMLWriter(fps=fps))
    plt.close(fig)
