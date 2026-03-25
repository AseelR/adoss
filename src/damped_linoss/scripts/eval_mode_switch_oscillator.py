import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(run_folder, example_idx=0):
    run_dir = BASE_DIR / run_folder
    inf_dir = run_dir / "inference"

    pred = np.asarray(load_pickle(inf_dir / "outputs_test.pkl"))[..., 0]
    truth = np.asarray(load_pickle(inf_dir / "truth_test.pkl"))[..., 0]
    inputs = np.asarray(load_pickle(inf_dir / "inputs_test.pkl"))

    data_dir = BASE_DIR / "damped_linoss" / "data" / "processed" / "mode_switch_oscillator"
    regime = np.asarray(load_pickle(data_dir / "regime_test.pkl"))

    sq_err = (pred - truth) ** 2
    results = {
        "overall_mse": float(np.mean(sq_err)),
        "retain_mse": float(np.mean(sq_err[regime == 0])),
        "process_mse": float(np.mean(sq_err[regime == 1])),
        "flush_mse": float(np.mean(sq_err[regime == 2])),
    }

    print(json.dumps(results, indent=2))
    with open(run_dir / "mode_switch_oscillator_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # qualitative plot
    x = inputs[example_idx]     # (T, 4)
    y = truth[example_idx]
    yhat = pred[example_idx]
    r = regime[example_idx]
    t = np.arange(len(y))

    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axs[0].plot(t, x[:, 0], label="forcing")
    axs[0].set_ylabel("forcing")
    axs[0].grid(True, alpha=0.25)

    axs[1].plot(t, r)
    axs[1].set_ylabel("regime")
    axs[1].grid(True, alpha=0.25)

    axs[2].plot(t, y, label="truth", linewidth=2)
    axs[2].plot(t, yhat, label="prediction", linewidth=2, alpha=0.85)
    axs[2].set_ylabel("output")
    axs[2].set_xlabel("time")
    axs[2].legend()
    axs[2].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(run_dir / "mode_switch_oscillator_example.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, required=True)
    parser.add_argument("--example_idx", type=int, default=0)
    args = parser.parse_args()
    main(args.run_folder, args.example_idx)