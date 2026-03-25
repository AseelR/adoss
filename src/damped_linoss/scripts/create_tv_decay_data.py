import numpy as np
import pickle
import os
from pathlib import Path

# damped_linoss/ directory
BASE_DIR = Path(__file__).resolve().parent.parent


def build_query_schedule(num_timesteps, mode="repeating"):
    """
    q_t in {0,1,2}:
      0 -> output long-memory latent
      1 -> output short-memory latent
      2 -> output zero / flush target
    """
    q = np.zeros(num_timesteps, dtype=np.int32)

    if mode == "repeating":
        lengths = [40, 25, 8, 25]     # long, short, flush, short
        states = [0, 1, 2, 1]
        t = 0
        i = 0
        while t < num_timesteps:
            seg_len = lengths[i % len(lengths)]
            seg_state = states[i % len(states)]
            q[t:min(t + seg_len, num_timesteps)] = seg_state
            t += seg_len
            i += 1
        return q

    elif mode == "piecewise":
        thirds = num_timesteps // 3
        q[thirds:2 * thirds] = 1
        q[2 * thirds:] = 2
        return q

    else:
        raise ValueError(f"Unknown mode {mode}")


def simulate_dual_memory(signal, q_seq):
    """
    Two latent states with incompatible timescales.
    Output is selected by q_t.
    """
    x_long = 0.0
    x_short = 0.0

    long_decay = 0.995
    short_decay = 0.20

    y = []
    x_long_hist = []
    x_short_hist = []

    for u_t, q_t in zip(signal, q_seq):
        x_long = long_decay * x_long + u_t
        x_short = short_decay * x_short + u_t

        if q_t == 0:
            y_t = x_long
        elif q_t == 1:
            y_t = x_short
        else:
            y_t = 0.0

        y.append(y_t)
        x_long_hist.append(x_long)
        x_short_hist.append(x_short)

    return (
        np.array(y, dtype=np.float32),
        np.array(x_long_hist, dtype=np.float32),
        np.array(x_short_hist, dtype=np.float32),
    )


if __name__ == "__main__":
    num_samples = 1000
    num_timesteps = 1000
    out_dir = BASE_DIR / "data" / "processed" / "synthetic_regression_tv"
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(0)

    inputs = []
    outputs = []
    queries = []
    long_latents = []
    short_latents = []

    for n in range(num_samples):
        signal = rng.normal(size=(num_timesteps,)).astype(np.float32)
        q_seq = build_query_schedule(num_timesteps, mode="repeating")

        y, x_long, x_short = simulate_dual_memory(signal, q_seq)

        # Input = [signal, one-hot query]
        q_onehot = np.eye(3, dtype=np.float32)[q_seq]               # (T, 3)
        x = np.concatenate([signal[:, None], q_onehot], axis=-1)    # (T, 4)

        inputs.append(x)
        outputs.append(y)
        queries.append(q_seq)
        long_latents.append(x_long)
        short_latents.append(x_short)

    inputs = np.array(inputs, dtype=np.float32)                  # (N, T, 4)
    outputs = np.array(outputs, dtype=np.float32)[..., None]     # (N, T, 1)
    queries = np.array(queries, dtype=np.int32)                  # (N, T)
    long_latents = np.array(long_latents, dtype=np.float32)      # (N, T)
    short_latents = np.array(short_latents, dtype=np.float32)    # (N, T)

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

    with open(out_dir / "query_test.pkl", "wb") as f:
        pickle.dump(queries[850:1000], f)
    with open(out_dir / "long_latent_test.pkl", "wb") as f:
        pickle.dump(long_latents[850:1000], f)
    with open(out_dir / "short_latent_test.pkl", "wb") as f:
        pickle.dump(short_latents[850:1000], f)

    print(f"Saved dual-timescale regression dataset to {out_dir}")