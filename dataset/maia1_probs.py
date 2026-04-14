import os
import chess
import numpy as np
import onnxruntime as ort
import torch
from concurrent.futures import ThreadPoolExecutor
from scipy.special import softmax
from tqdm import tqdm
from maia2 import inference, model as maia2_model
from maia2.utils import get_all_possible_moves

from .lcz_encoder import board_to_planes, uci_to_policy_idx

MAIA1_ELOS = [1100, 1300, 1500, 1700, 1900]
MAIA2_ELOS = [1100, 1300, 1500, 1700, 1900]
MAX_PLAYER_MOVES = 5
TOP_K = 5
MAIA1_BATCH_ROWS = 1024


def _checkpoint_paths(checkpoint_path):
    return {
        "probs": checkpoint_path,
        "policy_indices": checkpoint_path.replace(".npy", "_idx.npy"),
        "top5_probs": checkpoint_path.replace(".npy", "_top5p.npy"),
        "top5_indices": checkpoint_path.replace(".npy", "_top5i.npy"),
    }


def _load_checkpoint(checkpoint_path, result, policy_indices, top5_probs, top5_indices):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0

    checkpoint_paths = _checkpoint_paths(checkpoint_path)
    saved = np.load(checkpoint_paths["probs"])
    row_count = len(saved)
    result[:row_count] = saved

    if os.path.exists(checkpoint_paths["policy_indices"]):
        policy_indices[:row_count] = np.load(checkpoint_paths["policy_indices"])
    if os.path.exists(checkpoint_paths["top5_probs"]):
        top5_probs[:row_count] = np.load(checkpoint_paths["top5_probs"])
    if os.path.exists(checkpoint_paths["top5_indices"]):
        top5_indices[:row_count] = np.load(checkpoint_paths["top5_indices"])

    return row_count


def _save_checkpoint(checkpoint_path, row_count, result, policy_indices, top5_probs, top5_indices):
    if not checkpoint_path:
        return

    checkpoint_paths = _checkpoint_paths(checkpoint_path)
    np.save(checkpoint_paths["probs"], result[:row_count])
    np.save(checkpoint_paths["policy_indices"], policy_indices[:row_count])
    np.save(checkpoint_paths["top5_probs"], top5_probs[:row_count])
    np.save(checkpoint_paths["top5_indices"], top5_indices[:row_count])


def _clear_checkpoint(checkpoint_path):
    if not checkpoint_path:
        return

    for path in _checkpoint_paths(checkpoint_path).values():
        if os.path.exists(path):
            os.remove(path)


def _save_feature_cache(cache_dir, prefix, probs, policy_indices, top5_probs, top5_indices):
    np.save(os.path.join(cache_dir, f"{prefix}_probs.npy"), probs)
    np.save(os.path.join(cache_dir, f"{prefix}_policy_indices.npy"), policy_indices)
    np.save(os.path.join(cache_dir, f"{prefix}_top5_probs.npy"), top5_probs)
    np.save(os.path.join(cache_dir, f"{prefix}_top5_indices.npy"), top5_indices)


def _collect_puzzle_entries(fens, moves_col, row_start, row_end):
    entries = []
    for row_idx in range(row_start, row_end):
        board = chess.Board(fens[row_idx])
        moves_uci = str(moves_col[row_idx]).split()
        player_color = not board.turn
        player_move_idx = 0

        for uci in moves_uci:
            if player_move_idx >= MAX_PLAYER_MOVES:
                break
            if board.turn == player_color:
                entries.append((row_idx, player_move_idx, board.copy(stack=False), uci))
                player_move_idx += 1
            board.push(chess.Move.from_uci(uci))
    return entries


def _top_k_probs_and_indices(probabilities):
    top_k_unsorted = np.argpartition(probabilities, -TOP_K)[-TOP_K:]
    top_k_sorted = top_k_unsorted[np.argsort(probabilities[top_k_unsorted])[::-1]]
    return probabilities[top_k_sorted], top_k_sorted


def _load_sessions(models_dir, device=None):
    if device and device.startswith("cuda"):
        device_id = int(device.split(":")[1]) if ":" in device else 0
        providers = [('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    sessions = {}
    for elo in MAIA1_ELOS:
        path = os.path.join(models_dir, f"model{elo}.onnx")
        if os.path.exists(path):
            sessions[elo] = ort.InferenceSession(path, providers=providers)
    return sessions


def _run_elo_session(args):
    session, input_name, batch_planes = args
    logits = session.run(None, {input_name: batch_planes})[0]
    return softmax(logits, axis=1)


def compute_maia1_move_probs(df, models_dir="./maia1", checkpoint_path=None, device=None):
    sessions = _load_sessions(models_dir, device=device)
    if not sessions:
        raise FileNotFoundError(f"No Maia-1 ONNX models found in {models_dir}")

    available_elos = sorted(sessions.keys())
    input_name = next(iter(sessions.values())).get_inputs()[0].name
    n = len(df)
    n_elos = len(available_elos)
    result = np.zeros((n, MAX_PLAYER_MOVES, n_elos), dtype=np.float32)
    policy_indices = np.full((n, MAX_PLAYER_MOVES), -1, dtype=np.int32)
    top5_probs = np.zeros((n, MAX_PLAYER_MOVES, n_elos, TOP_K), dtype=np.float32)
    top5_indices = np.full((n, MAX_PLAYER_MOVES, n_elos, TOP_K), -1, dtype=np.int32)

    start_idx = _load_checkpoint(checkpoint_path, result, policy_indices, top5_probs, top5_indices)
    if start_idx:
        print(f"Resuming Maia-1 from row {start_idx}")

    fens = df["FEN"].tolist()
    moves_col = df["Moves"].tolist()

    with ThreadPoolExecutor(max_workers=n_elos) as executor:
        for chunk_start in tqdm(range(start_idx, n, MAIA1_BATCH_ROWS), desc="Maia-1 probs"):
            chunk_end = min(chunk_start + MAIA1_BATCH_ROWS, n)
            chunk_entries = _collect_puzzle_entries(fens, moves_col, chunk_start, chunk_end)

            entries = []
            for row_idx, player_move_idx, board, uci in chunk_entries:
                policy_idx = uci_to_policy_idx(board, uci)
                if policy_idx is None:
                    continue
                policy_indices[row_idx, player_move_idx] = policy_idx
                entries.append((row_idx, player_move_idx, board_to_planes(board), policy_idx))

            if not entries:
                continue

            batch_planes = np.stack([e[2] for e in entries])

            elo_probs_list = list(executor.map(
                _run_elo_session,
                [(sessions[elo], input_name, batch_planes) for elo in available_elos]
            ))

            for elo_idx, probs in enumerate(elo_probs_list):
                for i, (row_idx, player_move_idx, _, policy_idx) in enumerate(entries):
                    p = probs[i]
                    top_k_probs, top_k_indices = _top_k_probs_and_indices(p)
                    result[row_idx, player_move_idx, elo_idx] = p[policy_idx]
                    top5_probs[row_idx, player_move_idx, elo_idx] = top_k_probs
                    top5_indices[row_idx, player_move_idx, elo_idx] = top_k_indices

            if checkpoint_path and chunk_end % 10000 < MAIA1_BATCH_ROWS:
                _save_checkpoint(checkpoint_path, chunk_end, result, policy_indices, top5_probs, top5_indices)

    return result, policy_indices, top5_probs, top5_indices, available_elos


def build_maia1_features(df, models_dir="./maia1", data_file_name=None, cache_dir="./data"):
    if data_file_name is not None:
        cache_path = os.path.join(cache_dir, f"{data_file_name}_maia1_probs.npy")
        if os.path.exists(cache_path):
            return _derive_flat_features(np.load(cache_path))

    checkpoint_path = os.path.join(cache_dir, f"{data_file_name}_maia1_probs_ckpt.npy") if data_file_name else None
    probs, policy_indices, top5_probs, top5_indices, _ = compute_maia1_move_probs(df, models_dir=models_dir, checkpoint_path=checkpoint_path)

    if data_file_name is not None:
        _save_feature_cache(cache_dir, f"{data_file_name}_maia1", probs, policy_indices, top5_probs, top5_indices)
        _clear_checkpoint(checkpoint_path)

    return _derive_flat_features(probs)


def _derive_flat_features(probs):
    eps = 1e-7

    move_present = (probs > 0).astype(np.float32)
    safe_probs = np.where(move_present, probs, 1.0)
    joint_prob = safe_probs.prod(axis=1)

    log_probs = np.log(probs + eps) * move_present
    log_joint_prob = log_probs.sum(axis=1)
    ce = -log_probs

    return np.concatenate([
        probs.reshape(len(probs), -1),
        probs.mean(axis=1),
        probs.min(axis=1),
        probs.max(axis=1),
        ce.mean(axis=1),
        ce.max(axis=1),
        joint_prob,
        log_joint_prob,
    ], axis=1).astype(np.float32)


def _load_maia2_model(model_type="rapid", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = maia2_model.from_pretrained(type=model_type, device=device)
    model = model.to(device)
    model.eval()
    prepared = inference.prepare()
    all_moves_dict = {move: i for i, move in enumerate(get_all_possible_moves())}
    return model, prepared, all_moves_dict


def compute_maia2_move_probs(df, checkpoint_path=None, model_type="rapid", device=None):
    model, prepared, all_moves_dict = _load_maia2_model(model_type, device=device)

    n = len(df)
    n_elos = len(MAIA2_ELOS)
    result = np.zeros((n, MAX_PLAYER_MOVES, n_elos), dtype=np.float32)
    policy_indices = np.full((n, MAX_PLAYER_MOVES), -1, dtype=np.int32)
    top5_probs = np.zeros((n, MAX_PLAYER_MOVES, n_elos, TOP_K), dtype=np.float32)
    top5_indices = np.full((n, MAX_PLAYER_MOVES, n_elos, TOP_K), -1, dtype=np.int32)

    start_idx = _load_checkpoint(checkpoint_path, result, policy_indices, top5_probs, top5_indices)
    if start_idx:
        print(f"Resuming Maia-2 ({model_type}) from row {start_idx}")

    fens = df["FEN"].tolist()
    moves_col = df["Moves"].tolist()

    for row_idx in tqdm(range(start_idx, n), desc=f"Maia-2 {model_type} probs"):
        row_entries = _collect_puzzle_entries(fens, moves_col, row_idx, row_idx + 1)
        for _, player_move_idx, board, uci in row_entries:
            move_key = uci.rstrip("n")
            policy_idx = all_moves_dict.get(move_key)
            if policy_idx is None:
                continue

            policy_indices[row_idx, player_move_idx] = policy_idx
            fen = board.fen()

            for elo_idx, elo in enumerate(MAIA2_ELOS):
                move_probs, _ = inference.inference_each(model, prepared, fen, elo, elo)
                result[row_idx, player_move_idx, elo_idx] = move_probs.get(move_key, 0.0)

                sorted_moves = list(move_probs.items())[:TOP_K]
                top5_probs[row_idx, player_move_idx, elo_idx] = [prob for _, prob in sorted_moves]
                top5_indices[row_idx, player_move_idx, elo_idx] = [
                    all_moves_dict.get(predicted_move, -1) for predicted_move, _ in sorted_moves
                ]

        if checkpoint_path and (row_idx + 1) % 10000 == 0:
            _save_checkpoint(checkpoint_path, row_idx + 1, result, policy_indices, top5_probs, top5_indices)

    if checkpoint_path:
        _save_checkpoint(checkpoint_path, n, result, policy_indices, top5_probs, top5_indices)

    return result, policy_indices, top5_probs, top5_indices, MAIA2_ELOS


def build_maia2_features(df, data_file_name=None, cache_dir="./data"):
    if data_file_name is not None:
        cache_path = os.path.join(cache_dir, f"{data_file_name}_maia2_probs.npy")
        if os.path.exists(cache_path):
            return _derive_flat_features(np.load(cache_path))

    checkpoint_path = os.path.join(cache_dir, f"{data_file_name}_maia2_probs_ckpt.npy") if data_file_name else None
    probs, policy_indices, top5_probs, top5_indices, _ = compute_maia2_move_probs(df, checkpoint_path=checkpoint_path)

    if data_file_name is not None:
        _save_feature_cache(cache_dir, f"{data_file_name}_maia2", probs, policy_indices, top5_probs, top5_indices)
        _clear_checkpoint(checkpoint_path)

    return _derive_flat_features(probs)
