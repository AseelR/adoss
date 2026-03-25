import json
import pickle
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(run_folder):
    run_dir = BASE_DIR / run_folder
    inf_dir = run_dir / "inference"

    pred = np.asarray(load_pickle(inf_dir / "outputs_test.pkl"))[..., 0]
    truth = np.asarray(load_pickle(inf_dir / "truth_test.pkl"))[..., 0]

    data_dir = BASE_DIR / "damped_linoss" / "data" / "processed" / "write_hold_erase_query"
    cmd = np.asarray(load_pickle(data_dir / "cmd_test.pkl"))

    sq_err = (pred - truth) ** 2
    results = {
        "overall_mse": float(np.mean(sq_err)),
        "query_mse": float(np.mean(sq_err[cmd == 3])),
        "nonquery_mse": float(np.mean(sq_err[cmd != 3])),
    }

    print(json.dumps(results, indent=2))
    with open(run_dir / "write_hold_erase_query_metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, required=True)
    args = parser.parse_args()
    main(args.run_folder)