import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

MODELS_TO_TRAIN = [
    ["maia-1-1100", "maia-1-1300"],
    ["maia-1-1500", "maia-1-1700"],
    ["maia-1-1900"],
    ["maia-2-rapid-1100", "maia-2-rapid-1300"],
    ["maia-2-blitz-1100", "maia-2-blitz-1300"],
]

COMMON_ARGS = {
    "--csv_path": "../filtered.csv",
    "--stockfish_path": "../filtered_sf_evals.csv",
    "--themes_csv_path": "../filtered_themes_only.csv",
    "--data_dir": "./data",
    "--max_rows": "200000",
}

DEVICES = ["cuda:0", "cuda:1"]


def build_command(maia_source, device):
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "lightgbm_maia_specialist.py")]
    cmd += ["--maia_source", maia_source]
    cmd += ["--device", device]
    for k, v in COMMON_ARGS.items():
        cmd += [k, v]
    return cmd


def run_one(maia_source, device):
    print(f"[START] {maia_source} on {device}")
    result = subprocess.run(
        build_command(maia_source, device),
        cwd=os.path.dirname(__file__),
        capture_output=False,
        text=True,
    )
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[{status}] {maia_source} on {device}")
    return maia_source, result.returncode


if __name__ == "__main__":
    all_sources = [source for group in MODELS_TO_TRAIN for source in group]
    assigned = [(source, DEVICES[i % len(DEVICES)]) for i, source in enumerate(all_sources)]
    print(f"Training {len(all_sources)} specialist models (2 in parallel):\n")
    for source, device in assigned:
        print(f"  {source} -> {device}")
    print()

    failed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_one, source, device): source for source, device in assigned}
        for future in as_completed(futures):
            source, returncode = future.result()
            if returncode != 0:
                failed.append(source)

    print("\n--- Done ---")
    if failed:
        print(f"Failed models: {failed}")
    else:
        print("All models trained successfully.")
