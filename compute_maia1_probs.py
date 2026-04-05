import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from dataset.maia1_probs import compute_maia1_move_probs

csv_path = os.path.join(os.path.dirname(__file__), '..', 'filtered.csv')
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} puzzles from {csv_path}")

DEVICE = "cuda:0"

probs, available_elos = compute_maia1_move_probs(df, models_dir="./maia1", device=DEVICE)

out_path = os.path.join(os.path.dirname(__file__), 'data', 'filtered_maia1_probs.npy')
np.save(out_path, probs)
print(f"Saved probs shape {probs.shape} to {out_path}")
print(f"ELOs computed: {available_elos}")
print(f"Sample probs (first puzzle, all moves, all elos):\n{probs[0]}")
