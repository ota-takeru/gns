

from pathlib import Path

from sim.freefall import gen_freefall

if __name__ == "__main__":
    out_dir = Path("out/freefall")
    out_dir.mkdir(parents=True, exist_ok=True)

    pos, vel, gl = gen_freefall()
    import numpy as np
    npz_path = out_dir / "freefall.npz"
    np.savez(npz_path, pos=pos, vel=vel, g=np.array(gl["g"]), dt=np.array([gl["dt"]]))

    print(f"{npz_path} saved.")

    from viz.loaders import load_episode
    pos, vel, dt, g, rigid_id = load_episode(npz_path, include_rigid_id=True)
    print(f"pos.shape={pos.shape}, vel.shape={vel.shape}, dt={dt}, g={g}")

    from viz.plots import plot_time_series
    fig = plot_time_series(pos, vel)
    fig_path = out_dir / "freefall_timeseries.png"
    fig.savefig(fig_path, dpi=150)
    print(f"{fig_path} saved.")

    from viz.animate import animate_points
    mp4_path = out_dir / "freefall.mp4"
    animate_points(pos, save_mp4=mp4_path, rigid_id=rigid_id)
    print(f"{mp4_path} saved.")
