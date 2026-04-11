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

DEVICES = ["cuda"]


def build_command(maia_sources, blocks, device):
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "lightgbm_maia_specialist.py")]
    cmd += ["--maia_sources"] + maia_sources
    cmd += ["--blocks"] + blocks
    cmd += ["--device", device]
    for k, v in COMMON_ARGS.items():
        cmd += [k, v]
    return cmd


def run_one(label, maia_sources, blocks, device):
    print(f"[START] {label} on {device}  blocks={blocks}")
    result = subprocess.run(
        build_command(maia_sources, blocks, device),
        cwd=os.path.dirname(__file__),
        capture_output=False,
        text=True,
    )
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[{status}] {label}")
    return label, result.returncode


if __name__ == "__main__":
    jobs = []

    jobs.append(("all_blocks", MAIA_SOURCES, ALL_BLOCKS))

    for removed_block in ALL_BLOCKS:
        blocks = [b for b in ALL_BLOCKS if b != removed_block]
        label = f"no_{removed_block}"
        jobs.append((label, MAIA_SOURCES, blocks))

    assigned = [(label, sources, blocks, DEVICES[i % len(DEVICES)]) for i, (label, sources, blocks) in enumerate(jobs)]

    print(f"Running {len(assigned)} ablation jobs (2 in parallel):\n")
    for label, _, blocks, device in assigned:
        print(f"  {label:20s}  blocks={blocks}  -> {device}")
    print()

    failed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(run_one, label, sources, blocks, device): label
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
