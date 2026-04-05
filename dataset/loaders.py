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


def _compute_correct_move_rank(probs, top5_indices, policy_indices):
    top_k = top5_indices.shape[-1]
    pidx_expanded = policy_indices[:, :, np.newaxis, np.newaxis]
    matches = (top5_indices == pidx_expanded)
    has_match = matches.any(axis=-1)
    rank = np.argmax(matches, axis=-1) + 1
    rank = np.where(has_match, rank, top_k + 1).astype(np.float32)
    missing = (policy_indices == -1)[:, :, np.newaxis]
    return np.where(missing, float(top_k + 1), rank)


def _reduce_move_elo(arr):
    n = len(arr)
    return np.concatenate([
        arr.reshape(n, -1),
        arr.mean(axis=1),
        arr.min(axis=1),
        arr.max(axis=1),
    ], axis=1).astype(np.float32)


def _derive_maia2_extended_features(probs, top5_probs, top5_indices, policy_indices, move_ce=None, side_info_bce=None, value_output=None):
    from .maia1_probs import _derive_flat_features
    eps = 1e-7

    rank = _compute_correct_move_rank(probs, top5_indices, policy_indices)
    gap_to_top1 = top5_probs[:, :, :, 0] - probs
    prob_ratio = probs / (top5_probs[:, :, :, 0] + eps)

    feature_parts = [
        _derive_flat_features(probs),
        _reduce_move_elo(rank),
        _reduce_move_elo(gap_to_top1),
        _reduce_move_elo(prob_ratio),
    ]
    if move_ce is not None:
        feature_parts.append(_reduce_move_elo(move_ce))
    if side_info_bce is not None:
        feature_parts.append(_reduce_move_elo(side_info_bce))
    if value_output is not None:
        feature_parts.append(_reduce_move_elo(value_output))

    return np.concatenate(feature_parts, axis=1)


def load_maia2_features(data_file_name, data_dir="./data"):
    parts = []
    for model_type in ("rapid", "blitz"):
        probs_path = os.path.join(data_dir, f"{data_file_name}_maia2_{model_type}_probs.npy")
        top5p_path = os.path.join(data_dir, f"{data_file_name}_maia2_{model_type}_top5_probs.npy")
        top5i_path = os.path.join(data_dir, f"{data_file_name}_maia2_{model_type}_top5_indices.npy")
        pidx_path = os.path.join(data_dir, f"{data_file_name}_maia2_{model_type}_policy_indices.npy")
        ce_path = os.path.join(data_dir, f"{data_file_name}_maia2_{model_type}_move_ce.npy")
        bce_path = os.path.join(data_dir, f"{data_file_name}_maia2_{model_type}_side_info_bce.npy")
        val_path = os.path.join(data_dir, f"{data_file_name}_maia2_{model_type}_value.npy")

        if not os.path.exists(probs_path):
            print(f"Warning: {probs_path} not found, skipping {model_type} maia2 features")
            continue

        probs = np.load(probs_path)
        top5_probs = np.load(top5p_path)
        top5_indices = np.load(top5i_path)
        policy_indices = np.load(pidx_path)
        move_ce = np.load(ce_path) if os.path.exists(ce_path) else None
        side_info_bce = np.load(bce_path) if os.path.exists(bce_path) else None
        value_output = np.load(val_path) if os.path.exists(val_path) else None

        parts.append(_derive_maia2_extended_features(probs, top5_probs, top5_indices, policy_indices, move_ce, side_info_bce, value_output))

    return np.concatenate(parts, axis=1) if parts else None
