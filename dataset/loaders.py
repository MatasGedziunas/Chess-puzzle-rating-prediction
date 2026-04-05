import os

import numpy as np
import pandas as pd


def load_data(csv_path, stockfish_path=None, num_rows=None):
    df = pd.read_csv(csv_path)
    if num_rows:
        df = df.head(num_rows)

    if stockfish_path:
        sf_df = pd.read_csv(stockfish_path)
        df = df.merge(sf_df, on='PuzzleId', how='left')
        sf_cols = ['SF_Material', 'SF_Positional', 'SF_Final_Eval', 'SF_Last_Move_CP']
        stockfish_features = df[sf_cols].fillna(0).values.astype(np.float32)
    else:
        stockfish_features = None

    return df, stockfish_features


def load_stockfish_features(stockfish_evals_path, puzzles_df):
    sf_df = pd.read_csv(stockfish_evals_path)
    sf_df = sf_df[sf_df['PuzzleId'].isin(puzzles_df['PuzzleId'])]
    merged = puzzles_df[['PuzzleId']].merge(sf_df, on='PuzzleId', how='left')
    sf_cols = ['SF_Material', 'SF_Positional', 'SF_Final_Eval']
    return merged[sf_cols].fillna(0).values.astype(np.float32)
