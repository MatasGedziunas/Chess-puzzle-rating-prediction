import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from dataset.maia1_probs import compute_maia1_move_probs

DEVICE = "cuda:0"
CSV_PATH = "../filtered.csv"

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), CSV_PATH)
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    data_file_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(out_dir, f"{data_file_name}_maia1_probs.npy")
    ckpt_path = os.path.join(out_dir, f"{data_file_name}_maia1_probs_ckpt.npy")

    if os.path.exists(out_path):
        print(f"Output already exists at {out_path}, skipping.")
    else:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} puzzles from {csv_path}")

        probs, policy_indices, top5_probs, top5_indices, available_elos = compute_maia1_move_probs(
            df, models_dir="./maia1", checkpoint_path=ckpt_path, device=DEVICE
        )
        np.save(out_path, probs)
        np.save(out_path.replace("_probs.npy", "_policy_indices.npy"), policy_indices)
        np.save(out_path.replace("_probs.npy", "_top5_probs.npy"), top5_probs)
        np.save(out_path.replace("_probs.npy", "_top5_indices.npy"), top5_indices)
        if os.path.exists(ckpt_path):
            for suffix in [".npy", "_idx.npy", "_top5p.npy", "_top5i.npy"]:
                p = ckpt_path.replace(".npy", suffix) if suffix != ".npy" else ckpt_path
                if os.path.exists(p):
                    os.remove(p)
        print(f"Saved -> {out_path}  shape={probs.shape}  elos={available_elos}")

