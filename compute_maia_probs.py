import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from dataset.maia1_probs import compute_maia2_move_probs

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "../filtered.csv")
    out_dir = "./data"
    os.makedirs(out_dir, exist_ok=True)

    data_file_name = os.path.splitext(os.path.basename(csv_path))[0]

    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    for model_type in ("rapid", "blitz"):
        out_path = os.path.join(out_dir, f"{data_file_name}_maia2_{model_type}_probs.npy")
        ckpt_path = os.path.join(out_dir, f"{data_file_name}_maia2_{model_type}_probs_ckpt.npy")

        if os.path.exists(out_path):
            print(f"Output already exists at {out_path}, skipping.")
            continue

        probs, policy_indices, top5_probs, top5_indices, elos = compute_maia2_move_probs(df, checkpoint_path=ckpt_path, model_type=model_type)
        np.save(out_path, probs)
        np.save(out_path.replace("_probs.npy", "_policy_indices.npy"), policy_indices)
        np.save(out_path.replace("_probs.npy", "_top5_probs.npy"), top5_probs)
        np.save(out_path.replace("_probs.npy", "_top5_indices.npy"), top5_indices)
        if os.path.exists(ckpt_path):
            for suffix in [".npy", "_idx.npy", "_top5p.npy", "_top5i.npy"]:
                p = ckpt_path.replace(".npy", suffix) if suffix != ".npy" else ckpt_path
                if os.path.exists(p):
                    os.remove(p)
        print(f"Saved -> {out_path}  shape={probs.shape}  elos={elos}")
