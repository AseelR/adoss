import json
import pickle
from pathlib import Path

import yaml
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

BASE_DIR = Path(__file__).resolve().parent.parent.parent

from damped_linoss.data.create_dataset import create_dataset
from damped_linoss.models.create_model import create_model


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def squeeze_last(x):
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    return x


def load_run(run_folder):
    run_dir = BASE_DIR / run_folder

    with open(run_dir / "hyperparameters.yaml", "r") as f:
        hyperparameters = yaml.safe_load(f)

    seed = int(hyperparameters["seed"])
    dataset_key = jr.PRNGKey(seed)

    dataset = create_dataset(
        name=hyperparameters["dataset_name"],
        data_dir=hyperparameters["data_dir"],
        classification=hyperparameters["classification"],
        time_duration=hyperparameters["time_duration"] if hyperparameters["include_time"] else None,
        use_presplit=hyperparameters["use_presplit"],
        key=dataset_key,
    )

    hyperparameters |= {
        "input_dim": dataset.data_dim,
        "output_dim": dataset.label_dim,
    }

    empty_model, empty_state = create_model(hyperparameters, jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(run_dir / "model.eqx", empty_model)
    state = eqx.tree_deserialise_leaves(run_dir / "state.eqx", empty_state)
    model = eqx.tree_inference(model, True)

    return run_dir, hyperparameters, dataset, model, state


def predict_dataset(model, state, X, key):
    """
    Runs model on a full batch X of shape (N, T, D).
    """
    if getattr(model, "stateful", False) and getattr(model, "nondeterministic", False):
        yhat, _ = jax.vmap(model, in_axes=(0, None, 0))(X, state, jr.split(key, X.shape[0]))
    elif getattr(model, "stateful", False):
        yhat, _ = jax.vmap(model, in_axes=(0, None))(X, state)
    elif getattr(model, "nondeterministic", False):
        yhat = jax.vmap(model, in_axes=(0, 0))(X, jr.split(key, X.shape[0]))
    else:
        yhat = jax.vmap(model)(X)

    return yhat


def compute_steps_since_last_event(cmd, event_id):
    """
    cmd: (N, T)
    Returns array of shape (N, T), where entry is number of steps since
    the last event_id, or -1 if none has occurred yet.
    """
    N, T = cmd.shape
    out = np.full((N, T), -1, dtype=np.int32)

    for n in range(N):
        last = -1
        for t in range(T):
            if cmd[n, t] == event_id:
                last = t
            if last == -1:
                out[n, t] = -1
            else:
                out[n, t] = t - last

    return out


def safe_masked_mean(arr, mask):
    if np.any(mask):
        return float(np.mean(arr[mask]))
    return None


def main(run_folder, long_gap_threshold=50, post_erase_window=10):
    run_dir, hyperparameters, dataset, model, state = load_run(run_folder)

    test_loader = dataset.dataloaders["test"]
    X = jnp.asarray(test_loader.data)
    truth = squeeze_last(test_loader.labels)   # (N, T)

    pred = predict_dataset(model, state, X, jr.PRNGKey(0))
    pred = squeeze_last(pred)                  # (N, T)

    if pred.shape != truth.shape:
        raise ValueError(f"Prediction/truth mismatch: pred {pred.shape}, truth {truth.shape}")

    data_dir = BASE_DIR / "damped_linoss" / "data" / "processed" / "write_hold_erase_query"
    cmd = np.asarray(load_pickle(data_dir / "cmd_test.pkl"))      # (N, T)

    if cmd.shape != truth.shape:
        raise ValueError(f"Command/truth mismatch: cmd {cmd.shape}, truth {truth.shape}")

    sq_err = (pred - truth) ** 2

    query_mask = (cmd == 3)
    nonquery_mask = ~query_mask

    steps_since_write = compute_steps_since_last_event(cmd, event_id=1)
    steps_since_erase = compute_steps_since_last_event(cmd, event_id=2)

    long_gap_query_mask = query_mask & (steps_since_write >= long_gap_threshold)
    post_erase_query_mask = query_mask & (steps_since_erase >= 0) & (steps_since_erase <= post_erase_window)

    results = {
        "overall_mse": float(np.mean(sq_err)),
        "query_mse": safe_masked_mean(sq_err, query_mask),
        "nonquery_mse": safe_masked_mean(sq_err, nonquery_mask),
        f"long_gap_query_mse_ge_{long_gap_threshold}": safe_masked_mean(sq_err, long_gap_query_mask),
        f"post_erase_query_mse_le_{post_erase_window}": safe_masked_mean(sq_err, post_erase_query_mask),
        "num_query_points": int(np.sum(query_mask)),
        f"num_long_gap_queries_ge_{long_gap_threshold}": int(np.sum(long_gap_query_mask)),
        f"num_post_erase_queries_le_{post_erase_window}": int(np.sum(post_erase_query_mask)),
    }

    print(json.dumps(results, indent=2))

    with open(run_dir / "write_hold_erase_query_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(run_dir / "write_hold_erase_query_metrics.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, required=True)
    parser.add_argument("--long_gap_threshold", type=int, default=50)
    parser.add_argument("--post_erase_window", type=int, default=10)
    args = parser.parse_args()

    main(
        run_folder=args.run_folder,
        long_gap_threshold=args.long_gap_threshold,
        post_erase_window=args.post_erase_window,
    )

# import json
# import pickle
# from pathlib import Path
# import numpy as np

# BASE_DIR = Path(__file__).resolve().parent.parent.parent


# def load_pickle(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)


# def main(run_folder):
#     run_dir = BASE_DIR / run_folder
#     inf_dir = run_dir / "inference"

#     pred = np.asarray(load_pickle(inf_dir / "outputs_test.pkl"))[..., 0]
#     truth = np.asarray(load_pickle(inf_dir / "truth_test.pkl"))[..., 0]

#     data_dir = BASE_DIR / "damped_linoss" / "data" / "processed" / "write_hold_erase_query"
#     cmd = np.asarray(load_pickle(data_dir / "cmd_test.pkl"))

#     sq_err = (pred - truth) ** 2
#     results = {
#         "overall_mse": float(np.mean(sq_err)),
#         "query_mse": float(np.mean(sq_err[cmd == 3])),
#         "nonquery_mse": float(np.mean(sq_err[cmd != 3])),
#     }

#     print(json.dumps(results, indent=2))
#     with open(run_dir / "write_hold_erase_query_metrics.json", "w") as f:
#         json.dump(results, f, indent=2)


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--run_folder", type=str, required=True)
#     args = parser.parse_args()
#     main(args.run_folder)