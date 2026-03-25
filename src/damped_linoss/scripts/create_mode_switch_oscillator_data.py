import numpy as np
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt

# damped_linoss/ directory
BASE_DIR = Path(__file__).resolve().parent.parent


def build_regime_schedule(T, mode="repeating"):
    """
    r_t in {0,1,2}
      0 -> underdamped / retain
      1 -> critical-ish / process
      2 -> overdamped / flush
    """
    r = np.zeros(T, dtype=np.int32)

    if mode == "repeating":
        lengths = [70, 35, 14, 30]
        states = [0, 1, 2, 1]
        t = 0
        i = 0
        while t < T:
            seg_len = lengths[i % len(lengths)]
            seg_state = states[i % len(states)]
            r[t:min(t + seg_len, T)] = seg_state
            t += seg_len
            i += 1
        return r

    elif mode == "piecewise":
        thirds = T // 3
        r[thirds:2 * thirds] = 1
        r[2 * thirds:] = 2
        return r

    else:
        raise ValueError(f"Unknown mode {mode}")


def generate_forcing(T, rng, noise_std=0.08):
    t = np.arange(T, dtype=np.float32)
    u = (
        0.8 * np.sin(2 * np.pi * t / 45.0)
        + 0.35 * np.sin(2 * np.pi * t / 110.0 + 0.4)
        + noise_std * rng.normal(size=T)
    )
    return u.astype(np.float32)


def add_boundary_bursts(u, regime, rng, burst_scale=2.5, burst_width=2):
    u = u.copy()
    boundaries = np.where(regime[1:] != regime[:-1])[0] + 1
    for b in boundaries:
        amp = burst_scale * rng.normal()
        lo = max(0, b - burst_width)
        hi = min(len(u), b + burst_width + 1)
        u[lo:hi] += amp
    return u


def simulate_regime_switch_oscillator(u, regime):
    """
    Discrete 2nd-order oscillatory systems with different damping regimes.

    State update:
      x_{t+1} = x_t + v_{t+1}
      v_{t+1} = v_t + (-omega^2 x_t - g_r v_t + b u_t)

    Per-regime damping:
      regime 0: low damping  (retain / oscillatory)
      regime 1: moderate damping
      regime 2: strong damping (flush)

    Output = x_t
    """
    x = 0.0
    v = 0.0

    omega = 0.18
    b = 0.4

    g_map = {
        0: 0.05,   # underdamped-ish
        1: 0.35,   # medium damping
        2: 1.4,    # heavy damping / flush
    }

    xs = []
    vs = []
    gs = []

    for u_t, r_t in zip(u, regime):
        g = g_map[int(r_t)]
        v = v + (-omega**2 * x - g * v + b * u_t)
        x = x + v

        xs.append(x)
        vs.append(v)
        gs.append(g)

    return (
        np.array(xs, dtype=np.float32),
        np.array(vs, dtype=np.float32),
        np.array(gs, dtype=np.float32),
    )


def save_example_plot(out_dir, u, regime, x, v, g):
    t = np.arange(len(u))
    fig, axs = plt.subplots(4, 1, figsize=(11, 8), sharex=True)

    axs[0].plot(t, u)
    axs[0].set_ylabel("forcing")

    axs[1].plot(t, regime)
    axs[1].set_ylabel("regime")

    axs[2].plot(t, x, label="x")
    axs[2].plot(t, v, label="v", alpha=0.8)
    axs[2].legend()
    axs[2].set_ylabel("state")

    axs[3].plot(t, g)
    axs[3].set_ylabel("target damping")
    axs[3].set_xlabel("time")

    for ax in axs:
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_dir / "example_mode_switch_oscillator.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    num_samples = 1000
    num_timesteps = 1500
    out_dir = BASE_DIR / "data" / "processed" / "mode_switch_oscillator"
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(0)

    X = []
    Y = []
    R = []
    V = []
    G = []

    first_example_saved = False

    for _ in range(num_samples):
        regime = build_regime_schedule(num_timesteps, mode="repeating")
        u = generate_forcing(num_timesteps, rng, noise_std=0.08)
        u = add_boundary_bursts(u, regime, rng, burst_scale=2.5, burst_width=2)

        x, v, g = simulate_regime_switch_oscillator(u, regime)

        regime_onehot = np.eye(3, dtype=np.float32)[regime]        # (T, 3)
        inp = np.concatenate([u[:, None], regime_onehot], axis=-1) # (T, 4)
        out = x[:, None]                                            # (T, 1)

        X.append(inp)
        Y.append(out)
        R.append(regime)
        V.append(v)
        G.append(g)

        if not first_example_saved:
            save_example_plot(out_dir, u, regime, x, v, g)
            first_example_saved = True

    X = np.asarray(X, dtype=np.float32)  # (N, T, 4)
    Y = np.asarray(Y, dtype=np.float32)  # (N, T, 1)
    R = np.asarray(R, dtype=np.int32)    # (N, T)
    V = np.asarray(V, dtype=np.float32)  # (N, T)
    G = np.asarray(G, dtype=np.float32)  # (N, T)

    with open(out_dir / "X_train.pkl", "wb") as f:
        pickle.dump(X[0:700], f)
    with open(out_dir / "y_train.pkl", "wb") as f:
        pickle.dump(Y[0:700], f)

    with open(out_dir / "X_val.pkl", "wb") as f:
        pickle.dump(X[700:850], f)
    with open(out_dir / "y_val.pkl", "wb") as f:
        pickle.dump(Y[700:850], f)

    with open(out_dir / "X_test.pkl", "wb") as f:
        pickle.dump(X[850:1000], f)
    with open(out_dir / "y_test.pkl", "wb") as f:
        pickle.dump(Y[850:1000], f)

    with open(out_dir / "regime_test.pkl", "wb") as f:
        pickle.dump(R[850:1000], f)
    with open(out_dir / "velocity_test.pkl", "wb") as f:
        pickle.dump(V[850:1000], f)
    with open(out_dir / "target_damping_test.pkl", "wb") as f:
        pickle.dump(G[850:1000], f)

    print(f"Saved ModeSwitchOscillator dataset to {out_dir}")