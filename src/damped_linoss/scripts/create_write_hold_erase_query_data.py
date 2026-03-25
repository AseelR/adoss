import numpy as np
import pickle
import os
from pathlib import Path

# damped_linoss/ directory
BASE_DIR = Path(__file__).resolve().parent.parent


def sample_event_schedule(T, rng, p_write=0.003, p_erase=0.0015, p_query=0.01, min_gap=12):
    """
    Event channel semantics:
      cmd=0 -> none
      cmd=1 -> write
      cmd=2 -> erase
      cmd=3 -> query
    """
    cmd = np.zeros(T, dtype=np.int32)
    last_event = -10**9

    for t in range(T):
        if t - last_event < min_gap:
            continue

        r = rng.random()
        if r < p_write:
            cmd[t] = 1
            last_event = t
        elif r < p_write + p_erase:
            cmd[t] = 2
            last_event = t
        elif r < p_write + p_erase + p_query:
            cmd[t] = 3
            last_event = t

    return cmd


def generate_nuisance(T, rng, ar=0.985, noise_std=0.08, shock_prob=0.01, shock_scale=2.0):
    z = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        z[t] = ar * z[t - 1] + noise_std * rng.normal()
        if rng.random() < shock_prob:
            z[t] += shock_scale * rng.normal()
    return z


def make_example(
    T,
    rng,
    p_write=0.003,
    p_erase=0.0015,
    p_query=0.01,
    min_gap=12,
    nuisance_std=0.08,
    shock_prob=0.01,
    shock_scale=2.0,
    value_noise_std=0.05,
):
    """
    Input channels:
      x[:, 0] = write flag
      x[:, 1] = erase flag
      x[:, 2] = query flag
      x[:, 3] = candidate value channel
      x[:, 4] = nuisance channel

    Target:
      y[:, 0] = stored value ONLY at query positions, else 0

    This forces repeated online memory control throughout the sequence.
    """
    cmd = sample_event_schedule(
        T, rng,
        p_write=p_write,
        p_erase=p_erase,
        p_query=p_query,
        min_gap=min_gap,
    )

    nuisance = generate_nuisance(
        T, rng,
        ar=0.985,
        noise_std=nuisance_std,
        shock_prob=shock_prob,
        shock_scale=shock_scale,
    )

    x = np.zeros((T, 5), dtype=np.float32)
    y = np.zeros((T, 1), dtype=np.float32)

    mem = 0.0

    erase_burst = np.zeros(T, dtype=np.float32)
    for t in range(T):
        if cmd[t] == 2:   # erase
            hi = min(T, t + 4)
            erase_burst[t:hi] += 2.5 * rng.normal()

    

    for t in range(T):

        # nuisance always present, plus extra contamination around erase events
        x[t, 4] = nuisance[t] + erase_burst[t]

        # candidate value channel has distracting background even when not writing
        x[t, 3] = 0.25 * nuisance[t] + 0.25 * erase_burst[t] + value_noise_std * rng.normal()

        if cmd[t] == 1:  # write
            x[t, 0] = 1.0
            val = rng.choice([-1.0, 1.0])
            x[t, 3] = val + value_noise_std * rng.normal()
            mem = val

        elif cmd[t] == 2:  # erase
            x[t, 1] = 1.0
            mem = 0.0

        elif cmd[t] == 3:  # query
            x[t, 2] = 1.0
            y[t, 0] = mem

        # else: no supervision at non-query timesteps (target stays 0)

    return x, y, cmd


if __name__ == "__main__":
    num_samples = 1000
    num_timesteps = 2000
    out_dir = BASE_DIR / "data" / "processed" / "write_hold_erase_query"
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(0)

    X = []
    Y = []
    CMD = []

    for _ in range(num_samples):
        x, y, cmd = make_example(
            T=num_timesteps,
            rng=rng,
            p_write=0.003,
            p_erase=0.0015,
            p_query=0.01,
            min_gap=12,
            nuisance_std=0.08,
            shock_prob=0.01,
            shock_scale=2.0,
            value_noise_std=0.05,
        )
        X.append(x)
        Y.append(y)
        CMD.append(cmd)

    X = np.asarray(X, dtype=np.float32)   # (N, T, 5)
    Y = np.asarray(Y, dtype=np.float32)   # (N, T, 1)
    CMD = np.asarray(CMD, dtype=np.int32) # (N, T)

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

    with open(out_dir / "cmd_test.pkl", "wb") as f:
        pickle.dump(CMD[850:1000], f)

    print(f"Saved WriteHoldEraseQuery dataset to {out_dir}")