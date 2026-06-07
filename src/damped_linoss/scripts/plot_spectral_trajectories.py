import os
import sys
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

from damped_linoss.models.create_model import create_model
from damped_linoss.data.create_dataset import create_dataset


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def make_piecewise_input(
    T=300,
    input_dim=1,
    retain_amp=0.05,
    process_amp=1.0,
    flush_amp=2.0,
):
    """
    Designed schedule:
      - low-amplitude retain region
      - moderate process region
      - high-amplitude flush region
    """
    x = np.zeros((T, input_dim), dtype=np.float32)

    thirds = T // 3
    t = np.arange(T)

    x[:thirds, 0] = retain_amp * np.sin(0.08 * t[:thirds])
    x[thirds : 2 * thirds, 0] = process_amp * np.sin(0.15 * t[thirds : 2 * thirds])
    x[2 * thirds :, 0] = flush_amp * np.sign(np.sin(0.12 * t[2 * thirds :]))

    return jnp.array(x)


def local_eigs_damped_imex1(a, g, dt):
    """
    Eigenvalues of the per-mode 2x2 local recurrence for DampedIMEX1Layer:
        S = 1 + dt*g
        M = [[1/S, -dt*a/S],
             [dt/S, 1 - dt^2*a/S]]
    """
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


def get_first_damped_layer(model):
    for block in model.blocks:
        layer = block.layer
        if layer.__class__.__name__ == "DampedIMEX1Layer":
            return layer
    raise ValueError("No DampedIMEX1Layer found in model.")


def extract_spectral_trajectory(model, input_sequence):
    """
    Returns a dict with:
      G_seq:     (T, P)
      eig1:      (T, P)
      eig2:      (T, P)
      A_diag:    (P,)
      dt:        (P,)
    """
    layer = get_first_damped_layer(model)

    if layer.damping_mode not in ["constant", "input"]:
        raise ValueError(
            f"This script currently expects damping_mode in ['constant','input'], got {layer.damping_mode}"
        )

    A_diag, dt = layer._project_A_dt(layer.A_diag, layer.dt)
    _, G_base = layer._project_G(layer.G_diag, A_diag, dt)

    hidden_input = jax.vmap(model.linear_encoder)(input_sequence)
    A_diag_eff, G_seq = layer._compute_G_seq(hidden_input, G_base, A_diag, dt)

    def step_eigs(g_t):
        lam1, lam2 = jax.vmap(local_eigs_damped_imex1)(A_diag_eff, g_t, dt)
        return lam1, lam2

    eig1, eig2 = jax.vmap(step_eigs)(G_seq)

    return {
        "A_diag": np.array(A_diag_eff),
        "dt": np.array(dt),
        "G_base": np.array(G_base),
        "G_seq": np.array(G_seq),
        "eig1": np.array(eig1),
        "eig2": np.array(eig2),
    }


def set_publication_style():
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 400,
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "lines.linewidth": 2.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_unit_disk_trajectories(
    results,
    out_dir,
    num_modes_to_plot=4,
    figure_title=None,
    model_label=None,
):
    os.makedirs(out_dir, exist_ok=True)

    eig1 = results["eig1"]
    eig2 = results["eig2"]
    _, P = eig1.shape
    m = min(num_modes_to_plot, P)

    theta = np.linspace(0, 2 * np.pi, 512)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    ax.plot(circle_x, circle_y, "k--", linewidth=1.2, alpha=0.8, label="Unit circle")

    # lock colors so solid+dotted companion share the same color
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i in range(m):
        color = color_cycle[i % len(color_cycle)]

        ax.plot(
            eig1[:, i].real,
            eig1[:, i].imag,
            color=color,
            linewidth=2.4,
            label=f"Mode {i + 1}",
        )
        ax.plot(
            eig2[:, i].real,
            eig2[:, i].imag,
            color=color,
            linewidth=2.0,
            alpha=0.55,
            linestyle=":",
            label="_nolegend_",
        )

        ax.scatter(
            eig1[0, i].real,
            eig1[0, i].imag,
            color=color,
            s=26,
            zorder=3,
            label="_nolegend_",
        )
        ax.scatter(
            eig1[-1, i].real,
            eig1[-1, i].imag,
            color=color,
            s=48,
            marker="x",
            linewidths=1.6,
            zorder=3,
            label="_nolegend_",
        )

    if figure_title is not None:
        title = figure_title
    elif model_label is not None:
        title = f"Dynamic spectral trajectories ({model_label})"
    else:
        title = "Dynamic spectral trajectories"

    ax.set_title(title, pad=16, fontsize=26)
    ax.set_xlabel(r"Real part of $\lambda$", fontsize=24)
    ax.set_ylabel(r"Imaginary part of $\lambda$", fontsize=24)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(loc="best", frameon=False, ncol=1, fontsize=18)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "unit_disk_trajectories.png"), bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "unit_disk_trajectories.pdf"), bbox_inches="tight")
    plt.close()

def plot_timeseries_panels(
    results,
    input_sequence,
    out_dir,
    mode_idx=0,
    figure_title=None,
    model_label=None,
):
    os.makedirs(out_dir, exist_ok=True)

    G_seq = results["G_seq"]
    eig1 = results["eig1"]
    eig2 = results["eig2"]
    t = np.arange(len(input_sequence))
    x = np.array(input_sequence)[:, 0]

    fig, axs = plt.subplots(4, 1, figsize=(12.5, 10.5), sharex=True)

    axs[0].plot(t, x)
    axs[0].set_ylabel("Input", fontsize=22)
    if figure_title is None:
        if model_label is None:
            axs[0].set_title(f"Input-dependent spectral dynamics", pad=12, fontsize=24)
        else:
            axs[0].set_title(f"{model_label}: input-dependent spectral dynamics", pad=12, fontsize=24)
    else:
        axs[0].set_title(figure_title, pad=10)

    axs[1].plot(t, G_seq[:, mode_idx])
    axs[1].set_ylabel(r"$G_k$", fontsize=22)

    axs[2].plot(t, np.abs(eig1[:, mode_idx]), label=r"$|\lambda_1|$", )
    axs[2].plot(t, np.abs(eig2[:, mode_idx]), alpha=0.75, label=r"$|\lambda_2|$")
    axs[2].set_ylabel("Magnitude", fontsize=22)
    axs[2].legend(frameon=False, loc="best", fontsize=18)

    axs[3].plot(t, np.angle(eig1[:, mode_idx]), label=r"$\arg(\lambda_1)$")
    axs[3].plot(t, np.angle(eig2[:, mode_idx]), alpha=0.75, label=r"$\arg(\lambda_2)$")
    axs[3].set_ylabel("Phase", fontsize=22)
    axs[3].set_xlabel("Time step", fontsize=22)
    axs[3].legend(frameon=False, loc="best", fontsize=18)

    for ax in axs:
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=18)

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"mode_{mode_idx}_timeseries.png")
    pdf_path = os.path.join(out_dir, f"mode_{mode_idx}_timeseries.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def compute_spectral_summary(results, input_sequence):
    eig1 = results["eig1"]
    G_seq = results["G_seq"]
    inp = np.abs(np.array(input_sequence)[:, 0])

    travel = np.abs(eig1[1:] - eig1[:-1]).sum(axis=0)
    mean_travel = float(np.mean(travel))

    corrs = []
    for i in range(G_seq.shape[1]):
        g = G_seq[:, i]
        if np.std(g) < 1e-8 or np.std(inp) < 1e-8:
            corrs.append(0.0)
        else:
            corrs.append(np.corrcoef(inp, g)[0, 1])
    mean_corr = float(np.mean(corrs))

    return {
        "mean_spectral_travel": mean_travel,
        "per_mode_travel": travel,
        "mean_input_G_corr": mean_corr,
        "per_mode_input_G_corr": np.array(corrs),
    }


def rebuild_model_from_run(save_dir, hyperparameters):
    dataset_key = jr.PRNGKey(int(hyperparameters["seed"]))

    dataset = create_dataset(
        name=hyperparameters["dataset_name"],
        data_dir=hyperparameters["data_dir"],
        classification=hyperparameters["classification"],
        time_duration=hyperparameters["time_duration"] if hyperparameters["include_time"] else None,
        use_presplit=hyperparameters["use_presplit"],
        key=dataset_key,
    )

    hyperparameters = dict(hyperparameters)
    hyperparameters["input_dim"] = dataset.data_dim
    hyperparameters["output_dim"] = dataset.label_dim

    model_key = jr.PRNGKey(0)
    empty_model, _ = create_model(hyperparameters, model_key)
    model = eqx.tree_deserialise_leaves(save_dir / "model.eqx", empty_model)
    model = eqx.tree_inference(model, True)
    return model, hyperparameters


def main(
    model_save_folder,
    T=300,
    num_modes_to_plot=4,
    figure_title=None,
    model_label=None,
):
    set_publication_style()

    save_dir = BASE_DIR / model_save_folder
    out_dir = save_dir / "spectral_trajectory_plots"
    os.makedirs(out_dir, exist_ok=True)

    with open(save_dir / "hyperparameters.yaml", "r") as f:
        hyperparameters = yaml.safe_load(f)

    model, hyperparameters = rebuild_model_from_run(save_dir, hyperparameters)

    input_dim = hyperparameters["input_dim"]
    u = make_piecewise_input(T=T, input_dim=input_dim)

    results = extract_spectral_trajectory(model, u)

    save_pickle(out_dir / "spectral_trajectory.pkl", results)
    save_pickle(out_dir / "input_schedule.pkl", np.array(u))

    summary = compute_spectral_summary(results, u)
    save_pickle(out_dir / "spectral_summary.pkl", summary)

    with open(out_dir / "spectral_summary.txt", "w") as f:
        for k, v in summary.items():
            if np.isscalar(v):
                f.write(f"{k}: {v}\n")

    plot_unit_disk_trajectories(
        results,
        out_dir,
        num_modes_to_plot=num_modes_to_plot,
        figure_title=figure_title,
        model_label=model_label,
    )

    for i in range(min(num_modes_to_plot, results["G_seq"].shape[1])):
        plot_timeseries_panels(
            results,
            u,
            out_dir,
            mode_idx=i,
            figure_title=None,
            model_label=model_label,
        )

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_save_folder",
        type=str,
        required=True,
        help="Path to run folder containing model.eqx and hyperparameters.yaml",
    )
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--num_modes_to_plot", type=int, default=4)
    parser.add_argument(
        "--figure_title",
        type=str,
        default=None,
        help="Optional custom title for the unit-disk plot",
    )
    parser.add_argument(
        "--model_label",
        type=str,
        default=None,
        help="Optional short model label to include in titles",
    )
    args = parser.parse_args()

    main(
        model_save_folder=args.model_save_folder,
        T=args.T,
        num_modes_to_plot=args.num_modes_to_plot,
        figure_title=args.figure_title,
        model_label=args.model_label,
    )


# import os
# import sys
# import pickle
# from pathlib import Path
# import yaml


# import numpy as np
# import matplotlib.pyplot as plt

# import jax
# import jax.numpy as jnp
# import jax.random as jr
# import equinox as eqx

# BASE_DIR = Path(__file__).resolve().parent.parent.parent
# sys.path.append(str(BASE_DIR))

# from damped_linoss.models.create_model import create_model


# def load_pickle(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)


# def save_pickle(path, obj):
#     with open(path, "wb") as f:
#         pickle.dump(obj, f)


# def make_piecewise_input(
#     T=300,
#     input_dim=1,
#     retain_amp=0.05,
#     process_amp=1.0,
#     flush_amp=2.0,
# ):
#     """
#     Designed schedule:
#       - low-amplitude retain region
#       - moderate process region
#       - high-amplitude flush region
#       - repeat
#     """
#     x = np.zeros((T, input_dim), dtype=np.float32)

#     thirds = T // 3
#     t = np.arange(T)

#     # retain
#     x[:thirds, 0] = retain_amp * np.sin(0.08 * t[:thirds])

#     # process
#     x[thirds:2*thirds, 0] = process_amp * np.sin(0.15 * t[thirds:2*thirds])

#     # flush
#     x[2*thirds:, 0] = flush_amp * np.sign(np.sin(0.12 * t[2*thirds:]))

#     return jnp.array(x)


# def local_eigs_damped_imex1(a, g, dt):
#     """
#     Eigenvalues of the per-mode 2x2 local recurrence for DampedIMEX1Layer:
#         S = 1 + dt*g
#         M = [[1/S, -dt*a/S],
#              [dt/S, 1 - dt^2*a/S]]
#     """
#     s = 1.0 + dt * g
#     m11 = 1.0 / s
#     m12 = -(dt * a) / s
#     m21 = dt / s
#     m22 = 1.0 - (dt**2 * a) / s

#     tr = m11 + m22
#     det = m11 * m22 - m12 * m21
#     disc = tr**2 - 4.0 * det
#     sqrt_disc = jnp.sqrt(disc.astype(jnp.complex64))
#     lam1 = 0.5 * (tr + sqrt_disc)
#     lam2 = 0.5 * (tr - sqrt_disc)
#     return lam1, lam2


# def get_first_damped_layer(model):
#     for block in model.blocks:
#         layer = block.layer
#         if layer.__class__.__name__ == "DampedIMEX1Layer":
#             return layer
#     raise ValueError("No DampedIMEX1Layer found in model.")


# def extract_spectral_trajectory(model, input_sequence):
#     """
#     Returns a dict with:
#       G_seq:     (T, P)
#       eig1:      (T, P)
#       eig2:      (T, P)
#       A_diag:    (P,)
#       dt:        (P,)
#     """
#     layer = get_first_damped_layer(model)

#     if layer.damping_mode not in ["constant", "input"]:
#         raise ValueError(
#             f"This script currently expects damping_mode in ['constant','input'], got {layer.damping_mode}"
#         )

#     # Static projections exactly as in the layer
#     A_diag, dt = layer._project_A_dt(layer.A_diag, layer.dt)
#     _, G_base = layer._project_G(layer.G_diag, A_diag, dt)

#     # Build per-step G sequence
#     hidden_input = jax.vmap(model.linear_encoder)(input_sequence)
#     A_diag_eff, G_seq = layer._compute_G_seq(hidden_input, G_base, A_diag, dt)

#     # Compute local eigenvalues for each timestep and mode
#     def step_eigs(g_t):
#         lam1, lam2 = jax.vmap(local_eigs_damped_imex1)(A_diag_eff, g_t, dt)
#         return lam1, lam2

#     eig1, eig2 = jax.vmap(step_eigs)(G_seq)

#     return {
#         "A_diag": np.array(A_diag_eff),
#         "dt": np.array(dt),
#         "G_base": np.array(G_base),
#         "G_seq": np.array(G_seq),
#         "eig1": np.array(eig1),
#         "eig2": np.array(eig2),
#     }


# def plot_unit_disk_trajectories(results, out_dir, num_modes_to_plot=4):
#     os.makedirs(out_dir, exist_ok=True)

#     eig1 = results["eig1"]   # (T, P)
#     eig2 = results["eig2"]
#     T, P = eig1.shape
#     m = min(num_modes_to_plot, P)

#     theta = np.linspace(0, 2*np.pi, 512)
#     circle_x = np.cos(theta)
#     circle_y = np.sin(theta)

#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.plot(circle_x, circle_y, "k--", linewidth=1, label="Unit disk")

#     for i in range(m):
#         ax.plot(
#             eig1[:, i].real,
#             eig1[:, i].imag,
#             linewidth=2,
#             label=f"Mode {i} (λ1)"
#         )
#         ax.plot(
#             eig2[:, i].real,
#             eig2[:, i].imag,
#             linewidth=2,
#             alpha=0.6,
#             linestyle=":"
#         )

#         # mark start/end
#         ax.scatter(eig1[0, i].real, eig1[0, i].imag, s=25)
#         ax.scatter(eig1[-1, i].real, eig1[-1, i].imag, s=40, marker="x")

#     ax.set_title("Dynamic reachable-spectrum trajectories")
#     ax.set_xlabel("Re(λ)")
#     ax.set_ylabel("Im(λ)")
#     ax.set_aspect("equal")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, "unit_disk_trajectories.png"), dpi=200)
#     plt.close()


# def plot_timeseries_panels(results, input_sequence, out_dir, mode_idx=0):
#     os.makedirs(out_dir, exist_ok=True)

#     G_seq = results["G_seq"]
#     eig1 = results["eig1"]
#     eig2 = results["eig2"]
#     t = np.arange(len(input_sequence))

#     fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

#     axs[0].plot(t, np.array(input_sequence)[:, 0], linewidth=2)
#     axs[0].set_ylabel("input")
#     axs[0].set_title("Input schedule and induced spectral motion")

#     axs[1].plot(t, G_seq[:, mode_idx], linewidth=2)
#     axs[1].set_ylabel(f"G_k (mode {mode_idx})")

#     axs[2].plot(t, np.abs(eig1[:, mode_idx]), linewidth=2, label="|λ1|")
#     axs[2].plot(t, np.abs(eig2[:, mode_idx]), linewidth=2, label="|λ2|", alpha=0.7)
#     axs[2].set_ylabel("|λ_k|")
#     axs[2].legend()

#     axs[3].plot(t, np.angle(eig1[:, mode_idx]), linewidth=2, label="arg(λ1)")
#     axs[3].plot(t, np.angle(eig2[:, mode_idx]), linewidth=2, label="arg(λ2)", alpha=0.7)
#     axs[3].set_ylabel("phase")
#     axs[3].set_xlabel("time")
#     axs[3].legend()

#     for ax in axs:
#         ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, f"mode_{mode_idx}_timeseries.png"), dpi=200)
#     plt.close()



# def compute_spectral_summary(results, input_sequence):
#     eig1 = results["eig1"]   # (T, P)
#     G_seq = results["G_seq"] # (T, P)
#     inp = np.abs(np.array(input_sequence)[:, 0])

#     travel = np.abs(eig1[1:] - eig1[:-1]).sum(axis=0)   # (P,)
#     mean_travel = float(np.mean(travel))

#     corrs = []
#     for i in range(G_seq.shape[1]):
#         g = G_seq[:, i]
#         if np.std(g) < 1e-8 or np.std(inp) < 1e-8:
#             corrs.append(0.0)
#         else:
#             corrs.append(np.corrcoef(inp, g)[0, 1])
#     mean_corr = float(np.mean(corrs))

#     return {
#         "mean_spectral_travel": mean_travel,
#         "per_mode_travel": travel,
#         "mean_input_G_corr": mean_corr,
#         "per_mode_input_G_corr": np.array(corrs),
#     }

# def main(
#     model_save_folder,
#     T=300,
#     num_modes_to_plot=4,
# ):
#     save_dir = BASE_DIR / model_save_folder
#     out_dir = save_dir / "spectral_trajectory_plots"
#     os.makedirs(out_dir, exist_ok=True)

#     # Load hyperparameters
#     with open(save_dir / "hyperparameters.yaml", "r") as f:
#         hyperparameters = yaml.safe_load(f)

#     # Create empty model and load trained weights
#     model_key = jr.PRNGKey(0)
#     empty_model, empty_state = create_model(hyperparameters, model_key)
#     model = eqx.tree_deserialise_leaves(save_dir / "model.eqx", empty_model)
#     model = eqx.tree_inference(model, True)

#     # Designed input
#     input_dim = hyperparameters["input_dim"]
#     u = make_piecewise_input(T=T, input_dim=input_dim)

#     # Extract trajectory
#     results = extract_spectral_trajectory(model, u)

    

#     # Save raw arrays for later use
#     save_pickle(out_dir / "spectral_trajectory.pkl", results)
#     save_pickle(out_dir / "input_schedule.pkl", np.array(u))

#     # Save summary values 
#     summary = compute_spectral_summary(results, u)
#     save_pickle(out_dir / "spectral_summary.pkl", summary)

#     with open(out_dir / "spectral_summary.txt", "w") as f:
#         for k, v in summary.items():
#             if np.isscalar(v):
#                 f.write(f"{k}: {v}\n") 

#     # Plots
#     plot_unit_disk_trajectories(results, out_dir, num_modes_to_plot=num_modes_to_plot)
#     for i in range(min(num_modes_to_plot, results["G_seq"].shape[1])):
#         plot_timeseries_panels(results, u, out_dir, mode_idx=i)

#     print(f"Saved plots to {out_dir}")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--model_save_folder",
#         type=str,
#         required=True,
#         help="Path to run folder containing model.eqx/state.eqx/hyperparameters.pkl",
#     )
#     parser.add_argument("--T", type=int, default=300)
#     parser.add_argument("--num_modes_to_plot", type=int, default=4)
#     args = parser.parse_args()

#     main(
#         model_save_folder=args.model_save_folder,
#         T=args.T,
#         num_modes_to_plot=args.num_modes_to_plot,
#     )