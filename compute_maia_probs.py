import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from dataset.maia1_probs import compute_maia2_move_probs

MODEL_TYPE = "rapid"
CSV_PATH = "../filtered.csv"

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), CSV_PATH)
    out_dir = "./data"
    os.makedirs(out_dir, exist_ok=True)

    data_file_name = os.path.splitext(os.path.basename(csv_path))[0]

    out_path = os.path.join(out_dir, f"{data_file_name}_maia2_{MODEL_TYPE}_probs.npy")
    ckpt_path = os.path.join(out_dir, f"{data_file_name}_maia2_{MODEL_TYPE}_probs_ckpt.npy")

    if os.path.exists(out_path):
        print(f"Output already exists at {out_path}, skipping.")
    else:
        print(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")

        probs, policy_indices, top5_probs, top5_indices, move_ce, side_info_bce, value_output, elos = compute_maia2_move_probs(df, checkpoint_path=ckpt_path, model_type=MODEL_TYPE)
        np.save(out_path, probs)
        np.save(out_path.replace("_probs.npy", "_policy_indices.npy"), policy_indices)
        np.save(out_path.replace("_probs.npy", "_top5_probs.npy"), top5_probs)
        np.save(out_path.replace("_probs.npy", "_top5_indices.npy"), top5_indices)
        np.save(out_path.replace("_probs.npy", "_move_ce.npy"), move_ce)
        np.save(out_path.replace("_probs.npy", "_side_info_bce.npy"), side_info_bce)
        np.save(out_path.replace("_probs.npy", "_value.npy"), value_output)
        if os.path.exists(ckpt_path):
            for suffix in [".npy", "_idx.npy", "_top5p.npy", "_top5i.npy", "_ce.npy", "_bce.npy", "_val.npy"]:
                p = ckpt_path.replace(".npy", suffix) if suffix != ".npy" else ckpt_path
                if os.path.exists(p):
                    os.remove(p)
        print(f"Saved -> {out_path}  shape={probs.shape}  elos={elos}")
