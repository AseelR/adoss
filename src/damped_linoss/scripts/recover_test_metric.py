import os
import yaml
import argparse
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from pathlib import Path

from damped_linoss.data.create_dataset import create_dataset
from damped_linoss.models.create_model import create_model
from damped_linoss.train import evaluate


BASE_DIR = Path(__file__).resolve().parent.parent.parent


def safe_load(data, key, dtype=None):
    val = data.get(key, None)
    if val is None:
        raise KeyError(f"Key {key} does not exist")
    if dtype is not None:
        val = dtype(val)
    return val


def recover_test_metric(run_folder: str, write_file: bool = True):
    run_folder = Path(run_folder)

    hyperparameters_path = run_folder / "hyperparameters.yaml"
    model_path = run_folder / "model.eqx"
    state_path = run_folder / "state.eqx"

    if not hyperparameters_path.exists():
        raise FileNotFoundError(f"Missing {hyperparameters_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing {state_path}")

    with open(hyperparameters_path, "r") as f:
        hyperparameters = yaml.safe_load(f)

    # Fix data_dir in case YAML uses a relative path that depends on cwd
    data_dir = hyperparameters.get("data_dir", "data")
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        # interpret relative to repo root, not current working directory
        data_dir = BASE_DIR / data_dir
    hyperparameters["data_dir"] = str(data_dir)

    seed = safe_load(hyperparameters, "seed", int)
    dataset_name = safe_load(hyperparameters, "dataset_name", str)
    model_name = safe_load(hyperparameters, "model_name", str)

    dataset_key, model_key, test_key = jr.split(jr.PRNGKey(seed), 3)

    print(f"Recreating dataset {dataset_name}")
    dataset = create_dataset(
        name=dataset_name,
        data_dir=safe_load(hyperparameters, "data_dir", str),
        classification=safe_load(hyperparameters, "classification", bool),
        time_duration=(
            safe_load(hyperparameters, "time_duration", float)
            if safe_load(hyperparameters, "include_time", bool)
            else None
        ),
        use_presplit=safe_load(hyperparameters, "use_presplit", bool),
        key=dataset_key,
    )

    print(f"Recreating empty model {model_name}")
    hyperparameters |= {
        "input_dim": dataset.data_dim,
        "output_dim": dataset.label_dim,
    }
    empty_model, empty_state = create_model(
        hyperparameters=hyperparameters,
        key=model_key,
    )

    print("Loading checkpoint")
    model = eqx.tree_deserialise_leaves(model_path, empty_model)
    state = eqx.tree_deserialise_leaves(state_path, empty_state)

    inference_model = eqx.tree_inference(model, value=True)
    batch_size = safe_load(hyperparameters, "batch_size", int)

    print("Evaluating on test split")
    test_iter = dataset.dataloaders["test"].loop_epoch(batch_size)
    test_metric = evaluate(inference_model, state, test_iter, test_key)

    test_metric = float(test_metric)
    print(f"Recovered test metric: {test_metric}")

    if write_file:
        out_path = run_folder / "test_metric.txt"
        with open(out_path, "w") as f:
            f.write(str(test_metric))
        print(f"Wrote {out_path}")

    return test_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_folder",
        type=str,
        required=True,
        help="Path to a single run folder containing hyperparameters.yaml, model.eqx, state.eqx",
    )
    parser.add_argument(
        "--no_write",
        action="store_true",
        help="If set, do not write test_metric.txt",
    )
    args = parser.parse_args()

    recover_test_metric(
        run_folder=args.run_folder,
        write_file=not args.no_write,
    )