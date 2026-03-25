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
    if x.ndim == 2 and x.shape[-1] == 1:
        x = x[..., 0]
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


def get_first_damped_layer(model):
    for block in model.blocks:
        layer = block.layer
        if layer.__class__.__name__ == "DampedIMEX1Layer":
            return layer
    raise ValueError("No DampedIMEX1Layer found in model.")


@eqx.filter_jit
def predict_one(model, x, state, key):
    if model.stateful and model.nondeterministic:
        out, _ = model(x, state, key)
    elif model.stateful:
        out, _ = model(x, state)
    elif model.nondeterministic:
        out = model(x, key)
    else:
        out = model(x)
    return out


def local_eigs_damped_imex1(a, g, dt):
    s = 1.0 + dt * g
    m11 = 1.0 / s
    m12 = -(dt * a) / s
    m21 = dt / s
    m22 = 1.0 - (dt**2 * a) / s

    tr = m11 + m22
    det = m11 * m22 - m12 * m21
    disc = tr**2 - 4.0 * det
    sqrt_disc = jnp.sqrt(disc.astype(jnp.complex64))
    lam1 = 0.5 * (tr + sqrt_disc)
    lam2 = 0.5 * (tr - sqrt_disc)
    return lam1, lam2


def extract_schedule(model, raw_input_sequence):
    layer = get_first_damped_layer(model)

    # actual input to first DampedIMEX1 layer
    hidden_input = jax.vmap(model.linear_encoder)(raw_input_sequence)

    A_diag, dt = layer._project_A_dt(layer.A_diag, layer.dt)
    _, G_base = layer._project_G(layer.G_diag, A_diag, dt)

    A_diag_eff, G_seq = layer._compute_G_seq(hidden_input, G_base, A_diag, dt)

    def step_eigs(g_t):
        lam1, lam2 = jax.vmap(local_eigs_damped_imex1)(A_diag_eff, g_t, dt)
        return lam1, lam2

    eig1, eig2 = jax.vmap(step_eigs)(G_seq)

    return {
        "A_diag": np.array(A_diag_eff),
        "dt": np.array(dt),
        "G_seq": np.array(G_seq),
        "eig1": np.array(eig1),
        "eig2": np.array(eig2),
    }


def compute_summary(results):
    eig1 = results["eig1"]      # (T, P)
    G_seq = results["G_seq"]    # (T, P)

    travel = np.abs(eig1[1:] - eig1[:-1]).sum(axis=0)
    g_var = np.var(G_seq, axis=0)

    return {
        "mean_spectral_travel": float(np.mean(travel)),
        "mean_G_variance": float(np.mean(g_var)),
        "per_mode_travel": travel,
        "per_mode_G_variance": g_var,
    }


def plot_unit_disk(results, out_path, num_modes_to_plot=4):
    eig1 = results["eig1"]
    eig2 = results["eig2"]
    T, P = eig1.shape
    m = min(num_modes_to_plot, P)

    theta = np.linspace(0, 2*np.pi, 512)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(circle_x, circle_y, "k--", linewidth=1.5, label="Unit disk")

    for i in range(m):
        ax.plot(eig1[:, i].real, eig1[:, i].imag, linewidth=2, label=f"Mode {i} (λ1)")
        ax.plot(eig2[:, i].real, eig2[:, i].imag, linewidth=2, alpha=0.5, linestyle=":")
        ax.scatter(eig1[0, i].real, eig1[0, i].imag, s=35)
        ax.scatter(eig1[-1, i].real, eig1[-1, i].imag, s=50, marker="x")

    ax.set_title("Dynamic reachable-spectrum trajectories")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_timeseries(raw_x, pred, truth, results, out_path, mode_idx=0):
    t = np.arange(len(raw_x))

    G_seq = results["G_seq"]
    eig1 = results["eig1"]
    eig2 = results["eig2"]

    pred = np.asarray(pred)
    truth = np.asarray(truth)

    fig, axs = plt.subplots(5, 1, figsize=(11, 11), sharex=True)

    # panel 1: all input channels
    for c in range(raw_x.shape[1]):
        axs[0].plot(t, raw_x[:, c], linewidth=1.5, label=f"input ch {c}")
    axs[0].set_ylabel("input")
    axs[0].set_title("Actual task input and induced spectral schedule")
    axs[0].legend(loc="upper right")

    # panel 2: outputs
    pred_sq = np.squeeze(pred)
    truth_sq = np.squeeze(truth)

    # Case 1: sequence -> sequence
    if pred_sq.ndim == 1 and truth_sq.ndim == 1 and pred_sq.shape[0] == len(t) and truth_sq.shape[0] == len(t):
        axs[1].plot(t, truth_sq, linewidth=2, label="truth")
        axs[1].plot(t, pred_sq, linewidth=2, alpha=0.85, label="prediction")
        axs[1].set_ylabel("output")
        axs[1].legend(loc="upper right")

    # Case 2: sequence prediction vs scalar target
    elif pred_sq.ndim == 1 and pred_sq.shape[0] == len(t):
        truth_scalar = float(np.ravel(truth_sq)[-1])
        pred_scalar = float(np.ravel(pred_sq)[-1])

        axs[1].plot(t, pred_sq, linewidth=2, alpha=0.85, label="prediction seq")
        axs[1].axhline(truth_scalar, linewidth=2, label="truth scalar")
        axs[1].axhline(pred_scalar, linewidth=2, alpha=0.7, linestyle="--", label="pred final")
        axs[1].set_ylabel("output")
        axs[1].legend(loc="upper right")

    # Case 3: scalar -> scalar
    else:
        truth_scalar = float(np.ravel(truth_sq)[-1])
        pred_scalar = float(np.ravel(pred_sq)[-1])

        axs[1].axhline(truth_scalar, linewidth=2, label="truth")
        axs[1].axhline(pred_scalar, linewidth=2, alpha=0.85, label="prediction")
        axs[1].set_ylabel("final output")
        axs[1].legend(loc="upper right")

    # panel 3: damping
    axs[2].plot(t, G_seq[:, mode_idx], linewidth=2)
    axs[2].set_ylabel(f"G_k (mode {mode_idx})")

    # panel 4: eigenvalue magnitude
    axs[3].plot(t, np.abs(eig1[:, mode_idx]), linewidth=2, label="|λ1|")
    axs[3].plot(t, np.abs(eig2[:, mode_idx]), linewidth=2, label="|λ2|", alpha=0.7)
    axs[3].set_ylabel("|λ_k|")
    axs[3].legend(loc="upper right")

    # panel 5: phase
    axs[4].plot(t, np.angle(eig1[:, mode_idx]), linewidth=2, label="arg(λ1)")
    axs[4].plot(t, np.angle(eig2[:, mode_idx]), linewidth=2, label="arg(λ2)", alpha=0.7)
    axs[4].set_ylabel("phase")
    axs[4].set_xlabel("time")
    axs[4].legend(loc="upper right")

    for ax in axs:
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(run_folder, split="test", example_idx=0, num_modes_to_plot=4, mode_idx=0):
    run_dir, hyperparameters, dataset, model, state = load_run(run_folder)

    loader = dataset.dataloaders[split]
    raw_inputs = np.asarray(loader.data)
    raw_truth = np.asarray(loader.labels)

    x = jnp.asarray(raw_inputs[example_idx])
    truth = squeeze_last(raw_truth[example_idx])

    pred = predict_one(model, x, state, jr.PRNGKey(0))
    pred = squeeze_last(pred)

    results = extract_schedule(model, x)
    summary = compute_summary(results)

    out_dir = run_dir / f"task_spectral_{split}_{example_idx:03d}"
    os.makedirs(out_dir, exist_ok=True)

    save_pickle(out_dir / "spectral_schedule.pkl", results)
    save_pickle(out_dir / "summary.pkl", summary)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "mean_spectral_travel": summary["mean_spectral_travel"],
                "mean_G_variance": summary["mean_G_variance"],
            },
            f,
            indent=2,
        )

    plot_unit_disk(results, out_dir / "unit_disk_trajectories.png", num_modes_to_plot=num_modes_to_plot)
    plot_timeseries(np.asarray(x), np.asarray(pred), np.asarray(truth), results,
                    out_dir / f"mode_{mode_idx}_timeseries.png", mode_idx=mode_idx)

    print(f"Saved spectral schedule plots to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--example_idx", type=int, default=0)
    parser.add_argument("--num_modes_to_plot", type=int, default=4)
    parser.add_argument("--mode_idx", type=int, default=0)
    args = parser.parse_args()

    main(
        run_folder=args.run_folder,
        split=args.split,
        example_idx=args.example_idx,
        num_modes_to_plot=args.num_modes_to_plot,
        mode_idx=args.mode_idx,
    )