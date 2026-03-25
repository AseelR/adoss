import numpy as np
import pickle
import os
from pathlib import Path

# damped_linoss/ directory
BASE_DIR = Path(__file__).resolve().parent.parent


def generate_nuisance_signal(T, rng, ar=0.97, noise_std=0.15, shock_prob=0.01, shock_scale=2.5):
    """
    Smooth nuisance / distractor process with occasional shocks.
    """
    z = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        z[t] = ar * z[t - 1] + noise_std * rng.normal()
        if rng.random() < shock_prob:
            z[t] += shock_scale * rng.normal()
    return z


def sample_event_times(T, rng, p_write=0.002, p_reset=0.001, min_gap=20):
    """
    Sample sparse write / reset events with minimum spacing.
    Returns:
        write_times: list[int]
        reset_times: list[int]
    """
    write_times = []
    reset_times = []
    last_event = -10**9

    for t in range(T):
        if t - last_event < min_gap:
            continue

        r = rng.random()
        if r < p_write:
            write_times.append(t)
            last_event = t
        elif r < p_write + p_reset:
            reset_times.append(t)
            last_event = t

    return write_times, reset_times


def make_example(
    T,
    rng,
    p_write=0.002,
    p_reset=0.001,
    min_gap=20,
    nuisance_noise_std=0.15,
    candidate_noise_std=0.08,
    shock_prob=0.01,
    shock_scale=2.5,
):
    """
    Input channels:
      x[:, 0] = write flag
      x[:, 1] = reset flag
      x[:, 2] = candidate value channel
      x[:, 3] = nuisance / distractor channel

    Target:
      y[:, 0] = current memory trace

    Semantics:
      - write flag means store candidate value
      - reset flag means clear memory to 0
      - otherwise hold current memory while nuisance continues
    """
    x = np.zeros((T, 4), dtype=np.float32)
    y = np.zeros((T, 1), dtype=np.float32)

    nuisance = generate_nuisance_signal(
        T,
        rng,
        ar=0.97,
        noise_std=nuisance_noise_std,
        shock_prob=shock_prob,
        shock_scale=shock_scale,
    )

    write_times, reset_times = sample_event_times(
        T, rng, p_write=p_write, p_reset=p_reset, min_gap=min_gap
    )
    write_set = set(write_times)
    reset_set = set(reset_times)

    mem = 0.0

    for t in range(T):
        # always present nuisance channel
        x[t, 3] = nuisance[t]

        # candidate channel has noisy distractor by default
        x[t, 2] = 0.35 * nuisance[t] + candidate_noise_std * rng.normal()

        # reset has priority if both somehow collide
        if t in reset_set:
            x[t, 1] = 1.0
            mem = 0.0

        elif t in write_set:
            x[t, 0] = 1.0
            val = rng.choice([-1.0, 1.0])
            x[t, 2] = val + candidate_noise_std * rng.normal()
            mem = val

        y[t, 0] = mem

    return x, y


if __name__ == "__main__":
    num_samples = 1000
    num_timesteps = 2000

    out_dir = BASE_DIR / "data" / "processed" / "write_hold_reset"
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(0)

    X = []
    Y = []

    for _ in range(num_samples):
        x, y = make_example(
            T=num_timesteps,
            rng=rng,
            p_write=0.002,
            p_reset=0.001,
            min_gap=20,
            nuisance_noise_std=0.15,
            candidate_noise_std=0.08,
            shock_prob=0.01,
            shock_scale=2.5,
        )
        X.append(x)
        Y.append(y)

    X = np.asarray(X, dtype=np.float32)   # (N, T, 4)
    Y = np.asarray(Y, dtype=np.float32)   # (N, T, 1)

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

    print(f"Saved WriteHoldReset dataset to {out_dir}")