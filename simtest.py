

from sim.freefall import gen_freefall

if __name__ == "__main__":
    pos, vel, gl = gen_freefall()
    import numpy as np
    np.savez("freefall.npz", pos=pos, vel=vel, g=np.array(gl["g"]), dt=np.array([gl["dt"]]))

    print("freefall.npz saved.")

    from viz.loaders import load_episode
    pos, vel, dt, g = load_episode("freefall.npz")
    print(f"pos.shape={pos.shape}, vel.shape={vel.shape}, dt={dt}, g={g}")
    from viz.plots import plot_time_series
    fig = plot_time_series(pos, vel)
    fig.savefig("freefall_timeseries.png", dpi=150)
    print("freefall_timeseries.png saved.")
    from viz.animate import animate_points
    animate_points(pos, save_mp4="freefall.mp4")
    print("freefall.mp4 saved.")