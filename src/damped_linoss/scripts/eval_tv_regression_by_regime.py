import os
import sys
import json
import pickle
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from damped_linoss.data.create_dataset import create_dataset
from damped_linoss.models.create_model import create_model


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def squeeze_last(x):
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    return x


def load_run(run_folder):
    run_dir = BASE_DIR / run_folder

    with open(run_dir / "hyperparameters.yaml", "r") as f:
        hyperparameters = yaml.safe_load(f)

    dataset = create_dataset(
        name=hyperparameters["dataset_name"],
        data_dir=hyperparameters["data_dir"],
        classification=hyperparameters["classification"],
        time_duration=hyperparameters["time_duration"] if hyperparameters["include_time"] else None,
        use_presplit=hyperparameters["use_presplit"],
        key=jr.PRNGKey(int(hyperparameters["seed"])),
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


@eqx.filter_jit
def batched_predict(model, inputs, state, key):
    if model.nondeterministic and model.stateful:
        out, _ = jax.vmap(model, in_axes=(0, None, None))(inputs, state, key)
    elif model.stateful:
        out, _ = jax.vmap(model, in_axes=(0, None))(inputs, state)
    elif model.nondeterministic:
        out = jax.vmap(model, in_axes=(0, None))(inputs, key)
    else:
        out = jax.vmap(model)(inputs)
    return out


def compute_regime_mse(pred, truth, query):
    sq_err = (pred - truth) ** 2
    results = {"overall_mse": float(np.mean(sq_err))}
    for q in [0, 1, 2]:
        mask = (query == q)
        results[f"regime_{q}_mse"] = float(np.mean(sq_err[mask]))
    return results


def shade_regimes(ax, query):
    colors = {
        0: ("#dff0d8", "retain / long"),
        1: ("#d9edf7", "short"),
        2: ("#f2dede", "flush"),
    }
    T = len(query)
    start = 0
    while start < T:
        q = query[start]
        end = start + 1
        while end < T and query[end] == q:
            end += 1
        ax.axvspan(start, end, color=colors[q][0], alpha=0.25)
        start = end


def plot_example(run_dir, inputs, pred, truth, query, example_idx=0):
    x = np.asarray(inputs[example_idx])      # (T, 4)
    yhat = np.asarray(pred[example_idx])     # (T,)
    y = np.asarray(truth[example_idx])       # (T,)
    q = np.asarray(query[example_idx])       # (T,)

    t = np.arange(len(y))

    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    # panel 1: signal
    axs[0].plot(t, x[:, 0], linewidth=1.8)
    shade_regimes(axs[0], q)
    axs[0].set_ylabel("signal")
    axs[0].set_title("TV regression example with regime shading")

    # panel 2: query channels
    axs[1].plot(t, x[:, 1], label="q=0", linewidth=1.5)
    axs[1].plot(t, x[:, 2], label="q=1", linewidth=1.5)
    axs[1].plot(t, x[:, 3], label="q=2", linewidth=1.5)
    axs[1].set_ylabel("query one-hot")
    axs[1].legend(loc="upper right")

    # panel 3: truth vs pred
    shade_regimes(axs[2], q)
    axs[2].plot(t, y, label="truth", linewidth=2)
    axs[2].plot(t, yhat, label="prediction", linewidth=2, alpha=0.85)
    axs[2].set_ylabel("output")
    axs[2].set_xlabel("time")
    axs[2].legend(loc="upper right")

    for ax in axs:
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(run_dir / "tv_regime_example.png", dpi=200)
    plt.close()


def main(run_folder, example_idx=0):
    run_dir, hyperparameters, dataset, model, state = load_run(run_folder)

    test_loader = dataset.dataloaders["test"]
    inputs = jnp.array(test_loader.data)
    truth = squeeze_last(test_loader.labels)

    pred = batched_predict(model, inputs, state, jr.PRNGKey(0))
    pred = squeeze_last(pred)

    data_dir = BASE_DIR / "damped_linoss" / "data" / "processed" / "synthetic_regression_tv"
    query = np.asarray(load_pickle(data_dir / "query_test.pkl"))

    if pred.shape != truth.shape:
        raise ValueError(f"Prediction/truth mismatch: {pred.shape} vs {truth.shape}")
    if pred.shape != query.shape:
        raise ValueError(f"Prediction/query mismatch: {pred.shape} vs {query.shape}")

    metrics = compute_regime_mse(pred, truth, query)

    print(json.dumps(metrics, indent=2))

    with open(run_dir / "tv_regime_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(run_dir / "tv_regime_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    plot_example(run_dir, inputs, pred, truth, query, example_idx=example_idx)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, required=True)
    parser.add_argument("--example_idx", type=int, default=0)
    args = parser.parse_args()

    main(args.run_folder, args.example_idx)