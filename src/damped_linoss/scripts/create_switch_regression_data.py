import numpy as np
import pickle
import os
from pathlib import Path

# damped_linoss/ directory
BASE_DIR = Path(__file__).resolve().parent.parent


def build_regime_schedule(num_timesteps, mode="repeating"):
    """
    r_t in {0,1,2}
      0 -> retain (long memory)
      1 -> process (short / medium memory)
      2 -> flush (reset / forget)
    """
    r = np.zeros(num_timesteps, dtype=np.int32)

    if mode == "repeating":
        lengths = [60, 30, 10, 25]   # retain, process, flush, process
        states = [0, 1, 2, 1]
        t = 0
        i = 0
        while t < num_timesteps:
            seg_len = lengths[i % len(lengths)]
            seg_state = states[i % len(states)]
            r[t:min(t + seg_len, num_timesteps)] = seg_state
            t += seg_len
            i += 1
        return r

    elif mode == "piecewise":
        thirds = num_timesteps // 3
        r[thirds:2 * thirds] = 1
        r[2 * thirds:] = 2
        return r

    else:
        raise ValueError(f"Unknown mode {mode}")


def generate_signal(num_timesteps, rng, noise_std=0.35):
    """
    Smooth-ish input with some oscillatory structure + noise.
    """
    t = np.arange(num_timesteps, dtype=np.float32)
    s = (
        0.8 * np.sin(2 * np.pi * t / 40.0)
        + 0.5 * np.sin(2 * np.pi * t / 125.0 + 0.7)
        + noise_std * rng.normal(size=num_timesteps)
    )
    return s.astype(np.float32)


def add_boundary_bursts(signal, regime, rng, burst_scale=3.0, burst_width=2):
    """
    Add shocks near regime boundaries to stress selective forgetting.
    """
    signal = signal.copy()
    boundaries = np.where(regime[1:] != regime[:-1])[0] + 1
    for b in boundaries:
        amp = burst_scale * rng.normal()
        lo = max(0, b - burst_width)
        hi = min(len(signal), b + burst_width + 1)
        signal[lo:hi] += amp
    return signal


def simulate_switch_system(signal, regime):
    """
    Three latent dynamics:
      x_long  : long memory
      x_mid   : medium memory / processing
      x_flush : aggressively damped

    Output depends on regime:
      retain  -> output x_long
      process -> output x_mid
      flush   -> output x_flush (near zero quickly)
    """
    x_long = 0.0
    x_mid = 0.0
    x_flush = 0.0

    long_decay = 0.995
    mid_decay = 0.75
    flush_decay = 0.05

    y = []
    x_long_hist = []
    x_mid_hist = []
    x_flush_hist = []

    for u_t, r_t in zip(signal, regime):
        x_long = long_decay * x_long + u_t
        x_mid = mid_decay * x_mid + u_t
        x_flush = flush_decay * x_flush + u_t

        if r_t == 0:
            y_t = x_long
        elif r_t == 1:
            y_t = x_mid
        else:
            y_t = x_flush

        y.append(y_t)
        x_long_hist.append(x_long)
        x_mid_hist.append(x_mid)
        x_flush_hist.append(x_flush)

    return (
        np.array(y, dtype=np.float32),
        np.array(x_long_hist, dtype=np.float32),
        np.array(x_mid_hist, dtype=np.float32),
        np.array(x_flush_hist, dtype=np.float32),
    )


if __name__ == "__main__":
    num_samples = 1000
    num_timesteps = 1200
    out_dir = BASE_DIR / "data" / "processed" / "synthetic_regression_switch"
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(0)

    inputs = []
    outputs = []
    regimes = []
    long_latents = []
    mid_latents = []
    flush_latents = []

    for n in range(num_samples):
        regime = build_regime_schedule(num_timesteps, mode="repeating")
        signal = generate_signal(num_timesteps, rng, noise_std=0.35)
        signal = add_boundary_bursts(signal, regime, rng, burst_scale=3.0, burst_width=2)

        y, x_long, x_mid, x_flush = simulate_switch_system(signal, regime)

        # input = [signal, one-hot regime]
        regime_onehot = np.eye(3, dtype=np.float32)[regime]         # (T, 3)
        x = np.concatenate([signal[:, None], regime_onehot], axis=-1)  # (T, 4)

        inputs.append(x)
        outputs.append(y)
        regimes.append(regime)
        long_latents.append(x_long)
        mid_latents.append(x_mid)
        flush_latents.append(x_flush)

    inputs = np.array(inputs, dtype=np.float32)                   # (N, T, 4)
    outputs = np.array(outputs, dtype=np.float32)[..., None]     # (N, T, 1)
    regimes = np.array(regimes, dtype=np.int32)                  # (N, T)
    long_latents = np.array(long_latents, dtype=np.float32)      # (N, T)
    mid_latents = np.array(mid_latents, dtype=np.float32)        # (N, T)
    flush_latents = np.array(flush_latents, dtype=np.float32)    # (N, T)

    with open(out_dir / "X_train.pkl", "wb") as f:
        pickle.dump(inputs[0:700], f)
    with open(out_dir / "y_train.pkl", "wb") as f:
        pickle.dump(outputs[0:700], f)

    with open(out_dir / "X_val.pkl", "wb") as f:
        pickle.dump(inputs[700:850], f)
    with open(out_dir / "y_val.pkl", "wb") as f:
        pickle.dump(outputs[700:850], f)

    with open(out_dir / "X_test.pkl", "wb") as f:
        pickle.dump(inputs[850:1000], f)
    with open(out_dir / "y_test.pkl", "wb") as f:
        pickle.dump(outputs[850:1000], f)

    with open(out_dir / "regime_test.pkl", "wb") as f:
        pickle.dump(regimes[850:1000], f)
    with open(out_dir / "long_latent_test.pkl", "wb") as f:
        pickle.dump(long_latents[850:1000], f)
    with open(out_dir / "mid_latent_test.pkl", "wb") as f:
        pickle.dump(mid_latents[850:1000], f)
    with open(out_dir / "flush_latent_test.pkl", "wb") as f:
        pickle.dump(flush_latents[850:1000], f)

    print(f"Saved switch-regression dataset to {out_dir}")