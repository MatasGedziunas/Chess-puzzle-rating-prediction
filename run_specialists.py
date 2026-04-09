import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

MODELS_TO_TRAIN = [
    ["maia-1-1100", "maia-1-1300", "maia-2-1100", "maia-2-1300"],
    ["maia-1-1500", "maia-2-1500"],
    ["maia-1-1700", "maia-1-1900", "maia-2-1700", "maia-2-1900"],
]

COMMON_ARGS = {
    "--csv_path": "../filtered.csv",
    "--stockfish_path": "../filtered_sf_evals.csv",
    "--themes_csv_path": "../filtered_themes_only.csv",
    "--data_dir": "./data",
    "--max_rows": "200000",
    "--splits_path": "./data/filtered_splits.npz",
}

DEVICES = ["cuda:0", "cuda:1"]


def build_command(maia_sources, device):
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "lightgbm_maia_specialist.py")]
    cmd += ["--maia_sources"] + maia_sources
    cmd += ["--device", device]
    for k, v in COMMON_ARGS.items():
        cmd += [k, v]
    return cmd


def run_one(maia_sources, device):
    label = " + ".join(maia_sources)
    print(f"[START] {label} on {device}")
    result = subprocess.run(
        build_command(maia_sources, device),
        cwd=os.path.dirname(__file__),
        capture_output=False,
        text=True,
    )
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[{status}] {label} on {device}")
    return label, result.returncode


if __name__ == "__main__":
    assigned = [(group, DEVICES[i % len(DEVICES)]) for i, group in enumerate(MODELS_TO_TRAIN)]
    print(f"Training {len(MODELS_TO_TRAIN)} specialist models (2 in parallel):\n")
    for group, device in assigned:
        print(f"  {' + '.join(group)} -> {device}")
    print()

    failed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_one, group, device): group for group, device in assigned}
        for future in as_completed(futures):
            label, returncode = future.result()
            if returncode != 0:
                failed.append(label)

    print("\n--- Done ---")
    if failed:
        print(f"Failed models: {failed}")
    else:
        print("All models trained successfully.")
