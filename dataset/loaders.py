import os

import numpy as np
import pandas as pd
from pathlib import Path

from .maia2_embeddings import process_puzzle_sequences


def load_data(csv_path, embeddings_path=None, stockfish_path=None, num_rows=None, load_maia_embeddings=False):
    df = pd.read_csv(csv_path)
    if num_rows:
        df = df.head(num_rows)

    if load_maia_embeddings:
        if embeddings_path and os.path.exists(embeddings_path):
            maia_embeddings = np.load(embeddings_path)
            if len(maia_embeddings) != len(df):
                raise ValueError("Embeddings length does not match dataset length")
        else:
            maia_embeddings = process_puzzle_sequences(df)

            dataset_name = Path(csv_path).stem
            save_dir = f"./data/{dataset_name}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/maia2.npy"
            np.save(save_path, maia_embeddings)
    else:
        maia_embeddings = None

    if stockfish_path:
        sf_df = pd.read_csv(stockfish_path)
        df = df.merge(sf_df, on='PuzzleId', how='left')
        sf_cols = ['SF_Material', 'SF_Positional', 'SF_Final_Eval', 'SF_Last_Move_CP']
        stockfish_features = df[sf_cols].fillna(0).values.astype(np.float32)
    else:
        stockfish_features = None

    return df, maia_embeddings, stockfish_features


def load_stockfish_features(stockfish_evals_path, puzzles_df):
    sf_df = pd.read_csv(stockfish_evals_path)
    sf_df = sf_df[sf_df['PuzzleId'].isin(puzzles_df['PuzzleId'])]
    merged = puzzles_df[['PuzzleId']].merge(sf_df, on='PuzzleId', how='left')
    sf_cols = ['SF_Material', 'SF_Positional', 'SF_Final_Eval']
    return merged[sf_cols].fillna(0).values.astype(np.float32)
