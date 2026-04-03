import os
import sys
import chess
import numpy as np
import onnxruntime as ort
import torch
from scipy.special import softmax
from tqdm import tqdm

from .lcz_encoder import board_to_planes, uci_to_policy_idx

MAIA1_ELOS = [1100, 1300, 1500, 1700, 1900]
MAIA2_ELOS = [1100, 1300, 1500, 1700, 1900]
MAX_PLAYER_MOVES = 5
TOP_K = 5


def _load_sessions(models_dir):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sessions = {}
    for elo in MAIA1_ELOS:
        path = os.path.join(models_dir, f"model{elo}.onnx")
        if os.path.exists(path):
            sessions[elo] = ort.InferenceSession(path, providers=providers)
    return sessions


def compute_maia1_move_probs(df, models_dir="./maia1", checkpoint_path=None):
    sessions = _load_sessions(models_dir)
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

    start_idx = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        saved = np.load(checkpoint_path)
        result[:len(saved)] = saved
        idx_path = checkpoint_path.replace(".npy", "_idx.npy")
        if os.path.exists(idx_path):
            policy_indices[:len(np.load(idx_path))] = np.load(idx_path)
        top5p_path = checkpoint_path.replace(".npy", "_top5p.npy")
        if os.path.exists(top5p_path):
            top5_probs[:len(np.load(top5p_path))] = np.load(top5p_path)
        top5i_path = checkpoint_path.replace(".npy", "_top5i.npy")
        if os.path.exists(top5i_path):
            top5_indices[:len(np.load(top5i_path))] = np.load(top5i_path)
        start_idx = len(saved)
        print(f"Resuming Maia-1 from row {start_idx}")

    for row_idx in tqdm(range(start_idx, n), desc="Maia-1 probs"):
        row = df.iloc[row_idx]
        board = chess.Board(row['FEN'])
        moves_uci = str(row['Moves']).split()
        player_color = not board.turn

        p_count = 0
        for uci in moves_uci:
            if p_count >= MAX_PLAYER_MOVES:
                break
            if board.turn == player_color:
                planes = board_to_planes(board)[np.newaxis]
                policy_idx = uci_to_policy_idx(board, uci)
                if policy_idx is not None:
                    policy_indices[row_idx, p_count] = policy_idx
                    for elo_idx, elo in enumerate(available_elos):
                        logits = sessions[elo].run(None, {input_name: planes})[0][0]
                        probs = softmax(logits)
                        result[row_idx, p_count, elo_idx] = probs[policy_idx]
                        tk_idx = np.argsort(probs)[-TOP_K:][::-1]
                        top5_probs[row_idx, p_count, elo_idx] = probs[tk_idx]
                        top5_indices[row_idx, p_count, elo_idx] = tk_idx
                p_count += 1
            board.push(chess.Move.from_uci(uci))

        if checkpoint_path and (row_idx + 1) % 10000 == 0:
            np.save(checkpoint_path, result[:row_idx + 1])
            np.save(checkpoint_path.replace(".npy", "_idx.npy"), policy_indices[:row_idx + 1])
            np.save(checkpoint_path.replace(".npy", "_top5p.npy"), top5_probs[:row_idx + 1])
            np.save(checkpoint_path.replace(".npy", "_top5i.npy"), top5_indices[:row_idx + 1])

    return result, policy_indices, top5_probs, top5_indices, available_elos


def build_maia1_features(df, models_dir="./maia1", data_file_name=None, cache_dir="./data"):
    if data_file_name is not None:
        cache_path = os.path.join(cache_dir, f"{data_file_name}_maia1_probs.npy")
        if os.path.exists(cache_path):
            return _derive_flat_features(np.load(cache_path))

    checkpoint_path = os.path.join(cache_dir, f"{data_file_name}_maia1_probs_ckpt.npy") if data_file_name else None
    probs, policy_indices, top5_probs, top5_indices, _ = compute_maia1_move_probs(df, models_dir=models_dir, checkpoint_path=checkpoint_path)

    if data_file_name is not None:
        np.save(cache_path, probs)
        np.save(os.path.join(cache_dir, f"{data_file_name}_maia1_policy_indices.npy"), policy_indices)
        np.save(os.path.join(cache_dir, f"{data_file_name}_maia1_top5_probs.npy"), top5_probs)
        np.save(os.path.join(cache_dir, f"{data_file_name}_maia1_top5_indices.npy"), top5_indices)
        if checkpoint_path and os.path.exists(checkpoint_path):
            for suffix in [".npy", "_idx.npy", "_top5p.npy", "_top5i.npy"]:
                p = checkpoint_path.replace(".npy", suffix) if suffix != ".npy" else checkpoint_path
                if os.path.exists(p):
                    os.remove(p)

    return _derive_flat_features(probs)


def _derive_flat_features(probs):
    eps = 1e-7
    ce = -np.log(probs + eps)

    move_present = (probs > 0).astype(np.float32)
    safe_probs = np.where(move_present, probs, 1.0)
    joint_prob = safe_probs.prod(axis=1)

    log_probs = np.log(probs + eps) * move_present
    log_joint_prob = log_probs.sum(axis=1)

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


def _load_maia2_model():
    from maia2 import model as maia2_model, utils as maia2_utils
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = maia2_model.from_pretrained(type="rapid", device=device)
    m.eval()
    elo_dict = maia2_utils.create_elo_dict()
    all_moves = maia2_utils.get_all_possible_moves()
    all_moves_dict = {mv: i for i, mv in enumerate(all_moves)}
    mirror_move = maia2_utils.mirror_move
    board_to_tensor = __import__("maia2.inference", fromlist=["board_to_tensor"]).board_to_tensor
    map_to_category = maia2_utils.map_to_category
    return m, device, elo_dict, all_moves_dict, mirror_move, board_to_tensor, map_to_category


def compute_maia2_move_probs(df, checkpoint_path=None):
    m, device, elo_dict, all_moves_dict, mirror_move, board_to_tensor, map_to_category = _load_maia2_model()

    n = len(df)
    n_elos = len(MAIA2_ELOS)
    result = np.zeros((n, MAX_PLAYER_MOVES, n_elos), dtype=np.float32)
    policy_indices = np.full((n, MAX_PLAYER_MOVES), -1, dtype=np.int32)
    top5_probs = np.zeros((n, MAX_PLAYER_MOVES, n_elos, TOP_K), dtype=np.float32)
    top5_indices = np.full((n, MAX_PLAYER_MOVES, n_elos, TOP_K), -1, dtype=np.int32)

    start_idx = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        saved = np.load(checkpoint_path)
        result[:len(saved)] = saved
        idx_path = checkpoint_path.replace(".npy", "_idx.npy")
        if os.path.exists(idx_path):
            policy_indices[:len(np.load(idx_path))] = np.load(idx_path)
        top5p_path = checkpoint_path.replace(".npy", "_top5p.npy")
        if os.path.exists(top5p_path):
            top5_probs[:len(np.load(top5p_path))] = np.load(top5p_path)
        top5i_path = checkpoint_path.replace(".npy", "_top5i.npy")
        if os.path.exists(top5i_path):
            top5_indices[:len(np.load(top5i_path))] = np.load(top5i_path)
        start_idx = len(saved)
        print(f"Resuming Maia-2 from row {start_idx}")

    elo_indices = [map_to_category(elo, elo_dict) for elo in MAIA2_ELOS]

    for row_idx in tqdm(range(start_idx, n), desc="Maia-2 probs"):
        row = df.iloc[row_idx]
        board = chess.Board(row['FEN'])
        moves_uci = str(row['Moves']).split()
        player_color = not board.turn

        p_count = 0
        for uci in moves_uci:
            if p_count >= MAX_PLAYER_MOVES:
                break
            if board.turn == player_color:
                is_black = not board.turn
                b = board.mirror() if is_black else board
                board_input = board_to_tensor(b).unsqueeze(0).to(device)
                move_key = mirror_move(uci) if is_black else uci
                move_key = move_key.rstrip("n")
                policy_idx = all_moves_dict.get(move_key)
                if policy_idx is not None:
                    policy_indices[row_idx, p_count] = policy_idx
                    for elo_idx, elo_cat in enumerate(elo_indices):
                        elo_t = torch.tensor([elo_cat]).to(device)
                        with torch.no_grad():
                            logits_policy, _, _ = m(board_input, elos_self=elo_t, elos_oppo=elo_t)
                        probs = torch.softmax(logits_policy[0], dim=0).cpu().numpy()
                        result[row_idx, p_count, elo_idx] = probs[policy_idx]
                        tk_idx = np.argsort(probs)[-TOP_K:][::-1]
                        top5_probs[row_idx, p_count, elo_idx] = probs[tk_idx]
                        top5_indices[row_idx, p_count, elo_idx] = tk_idx
                p_count += 1
            board.push(chess.Move.from_uci(uci))

        if checkpoint_path and (row_idx + 1) % 10000 == 0:
            np.save(checkpoint_path, result)
            np.save(checkpoint_path.replace(".npy", "_idx.npy"), policy_indices)
            np.save(checkpoint_path.replace(".npy", "_top5p.npy"), top5_probs)
            np.save(checkpoint_path.replace(".npy", "_top5i.npy"), top5_indices)

    if checkpoint_path:
        np.save(checkpoint_path, result)
        np.save(checkpoint_path.replace(".npy", "_idx.npy"), policy_indices)
        np.save(checkpoint_path.replace(".npy", "_top5p.npy"), top5_probs)
        np.save(checkpoint_path.replace(".npy", "_top5i.npy"), top5_indices)

    return result, policy_indices, top5_probs, top5_indices, MAIA2_ELOS


def build_maia2_features(df, data_file_name=None, cache_dir="./data"):
    if data_file_name is not None:
        cache_path = os.path.join(cache_dir, f"{data_file_name}_maia2_probs.npy")
        if os.path.exists(cache_path):
            return _derive_flat_features(np.load(cache_path))

    checkpoint_path = os.path.join(cache_dir, f"{data_file_name}_maia2_probs_ckpt.npy") if data_file_name else None
    probs, policy_indices, top5_probs, top5_indices, _ = compute_maia2_move_probs(df, checkpoint_path=checkpoint_path)

    if data_file_name is not None:
        np.save(cache_path, probs)
        np.save(os.path.join(cache_dir, f"{data_file_name}_maia2_policy_indices.npy"), policy_indices)
        np.save(os.path.join(cache_dir, f"{data_file_name}_maia2_top5_probs.npy"), top5_probs)
        np.save(os.path.join(cache_dir, f"{data_file_name}_maia2_top5_indices.npy"), top5_indices)
        if checkpoint_path and os.path.exists(checkpoint_path):
            for suffix in [".npy", "_idx.npy", "_top5p.npy", "_top5i.npy"]:
                p = checkpoint_path.replace(".npy", suffix) if suffix != ".npy" else checkpoint_path
                if os.path.exists(p):
                    os.remove(p)

    return _derive_flat_features(probs)
