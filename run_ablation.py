import argparse
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

ALL_BLOCKS = ["struct", "themes", "advanced", "stockfish", "maia1", "maia2"]

MAIA_SOURCES = ["maia-1-1100", "maia-1-1300", "maia-1-1500", "maia-1-1700", "maia-1-1900",
                "maia-2-1100", "maia-2-1300", "maia-2-1500", "maia-2-1700", "maia-2-1900"]

COMMON_ARGS = {
    "--csv_path": "../filtered.csv",
    "--stockfish_path": "../filtered_sf_evals.csv",
    "--themes_csv_path": "../filtered_themes_only.csv",
    "--data_dir": "./data",
    "--splits_path": "./data/filtered_splits.npz",
}

LIGHTGBM_DEVICES = ["cuda"]
CATBOOST_CUDA_DEVICES = [0]
MAX_WORKERS = 2


def build_command(trainer, maia_sources, blocks, device, max_rows):
    if trainer == "lightgbm":
        script_name = "lightgbm_maia_specialist.py"
    else:
        script_name = "train_catboost_full_dataset.py"

    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), script_name)]
    if trainer == "lightgbm":
        cmd += ["--maia_sources"] + maia_sources
    cmd += ["--blocks"] + blocks
    if trainer == "lightgbm":
        cmd += ["--device", device]
    else:
        cmd += ["--cuda_device", str(device)]
    if max_rows is not None:
        cmd += ["--max_rows", str(max_rows)]
    for k, v in COMMON_ARGS.items():
        cmd += [k, v]
    return cmd


def run_one(trainer, label, maia_sources, blocks, device, max_rows):
    print(f"[START] {label} trainer={trainer} device={device} blocks={blocks}")
    result = subprocess.run(
        build_command(trainer, maia_sources, blocks, device, max_rows),
        cwd=os.path.dirname(__file__),
        capture_output=False,
        text=True,
    )
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[{status}] {label}")
    return label, result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", choices=["lightgbm", "catboost"], default="lightgbm")
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    jobs = []

    jobs.append(("all_blocks", MAIA_SOURCES, ALL_BLOCKS))

    for removed_block in ALL_BLOCKS:
        blocks = [b for b in ALL_BLOCKS if b != removed_block]
        label = f"no_{removed_block}"
        jobs.append((label, MAIA_SOURCES, blocks))

    devices = LIGHTGBM_DEVICES if args.trainer == "lightgbm" else CATBOOST_CUDA_DEVICES
    assigned = [(label, sources, blocks, devices[i % len(devices)]) for i, (label, sources, blocks) in enumerate(jobs)]

    print(f"Running {len(assigned)} ablation jobs with trainer={args.trainer} ({MAX_WORKERS} in parallel):\n")
    for label, _, blocks, device in assigned:
        print(f"  {label:20s}  blocks={blocks}  -> {device}")
    print()

    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_one, args.trainer, label, sources, blocks, device, args.max_rows): label
            for label, sources, blocks, device in assigned
        }
        for future in as_completed(futures):
            label, returncode = future.result()
            if returncode != 0:
                failed.append(label)

    print("\n--- Done ---")
    if failed:
        print(f"Failed jobs: {failed}")
    else:
        print("All ablation jobs completed successfully.")
