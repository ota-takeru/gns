from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

DATASET_DIR = Path(__file__).resolve().parents[1] / "out" / "train"
SCENES = sorted(DATASET_DIR.glob("scene_*.npz"))
if not SCENES:
    raise FileNotFoundError(f"No scene npz files found under {DATASET_DIR}")

DATASET_PATH = SCENES[1]    #表示するシーンを選択
with np.load(DATASET_PATH) as data:
    pos = data["position"]
    rigid_id = data["rigid_id"]

fig, ax = plt.subplots()
col = rigid_id if len(np.unique(rigid_id)) > 1 else None
if col is None:
    sc = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=20)
else:
    sc = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=20, c=col, cmap="tab20")
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

    html_dir = DATASET_DIR / "animations"
    html_dir.mkdir(parents=True, exist_ok=True)
    html_path = html_dir / f"{DATASET_PATH.stem}.html"
    ani.save(html_path, writer=HTMLWriter(fps=100))
    print(f"Saved animation to {html_path}")
    plt.close(fig)
else:
    plt.show()
