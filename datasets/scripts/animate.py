from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

DATASET_PATH = Path(__file__).resolve().parents[1] / "out" / "train.h5"

with h5py.File(DATASET_PATH, "r") as f:
    pos = f["train/scene_000/positions"][:]

fig, ax = plt.subplots()
sc = ax.scatter([], [], s=10)
ax.set_xlim(pos[..., 0].min() - 1, pos[..., 0].max() + 1)
ax.set_ylim(pos[..., 1].min() - 1, pos[..., 1].max() + 1)
ax.set_aspect("equal")


def update(frame):
    sc.set_offsets(pos[frame])
    ax.set_title(f"frame={frame}")
    return (sc,)


ani = animation.FuncAnimation(fig, update, frames=len(pos), interval=20, blit=True)
backend = matplotlib.get_backend().lower()

if "agg" in backend:
    # 非対話バックエンドではHTMLを保存して確認できるようにする
    from matplotlib.animation import HTMLWriter

    html_path = DATASET_PATH.parent / "train_animation.html"
    ani.save(html_path, writer=HTMLWriter(fps=100))
    print(f"Saved animation to {html_path}")
    plt.close(fig)
else:
    plt.show()
