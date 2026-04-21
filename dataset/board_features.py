import os
import chess
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def _count_material(board, color):
    return sum(
        len(board.pieces(pt, color)) * PIECE_VALUES[pt]
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    )


def _piece_mobility(board, piece_type, color):
    return sum(
        1 for m in board.legal_moves
        if (p := board.piece_at(m.from_square)) and p.piece_type == piece_type and p.color == color
    )


def _attackers_near_king(board, king_color, attacking_color):
    king_square = board.king(king_color)
    if king_square is None:
        return 0
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    count = 0
    for file_offset in (-1, 0, 1):
        for rank_offset in (-1, 0, 1):
            target_file = king_file + file_offset
            target_rank = king_rank + rank_offset
            if 0 <= target_file <= 7 and 0 <= target_rank <= 7:
                count += len(board.attackers(attacking_color, chess.square(target_file, target_rank)))
    return count


def _pawn_islands(board, color):
    pawn_files = sorted({chess.square_file(square) for square in board.pieces(chess.PAWN, color)})
    if not pawn_files:
        return 0
    island_count = 1
    for index in range(1, len(pawn_files)):
        if pawn_files[index] - pawn_files[index - 1] > 1:
            island_count += 1
    return island_count


def _doubled_pawns(board, color):
    pawn_counts_by_file = [0] * 8
    for square in board.pieces(chess.PAWN, color):
        pawn_counts_by_file[chess.square_file(square)] += 1
    return sum(file_count - 1 for file_count in pawn_counts_by_file if file_count > 1)


def _isolated_pawns(board, color):
    pawn_files = {chess.square_file(square) for square in board.pieces(chess.PAWN, color)}
    return sum(
        1
        for square in board.pieces(chess.PAWN, color)
        if (pawn_file := chess.square_file(square)) and (pawn_file - 1) not in pawn_files and (pawn_file + 1) not in pawn_files
    )


def _passed_pawns(board, color):
    opponent_color = not color
    direction = 1 if color == chess.WHITE else -1
    count = 0
    for square in board.pieces(chess.PAWN, color):
        pawn_file = chess.square_file(square)
        pawn_rank = chess.square_rank(square)
        is_passed = True
        for candidate_file in (pawn_file - 1, pawn_file, pawn_file + 1):
            if 0 <= candidate_file <= 7:
                candidate_rank = pawn_rank + direction
                while 0 <= candidate_rank <= 7:
                    piece = board.piece_at(chess.square(candidate_file, candidate_rank))
                    if piece and piece.piece_type == chess.PAWN and piece.color == opponent_color:
                        is_passed = False
                        break
                    candidate_rank += direction
                if not is_passed:
                    break
        if is_passed:
            count += 1
    return count


def _undefended_pieces(board, color):
    cnt, val = 0, 0
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for square in board.pieces(piece_type, color):
            if not board.attackers(color, square):
                cnt += 1
                val += PIECE_VALUES[piece_type]
    return cnt, val


def _over_under_defended(board, color):
    opponent_color = not color
    over, under = 0, 0
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for square in board.pieces(piece_type, color):
            defender_count = len(board.attackers(color, square))
            attacker_count = len(board.attackers(opponent_color, square))
            if defender_count > attacker_count and attacker_count > 0:
                over += 1
            elif attacker_count > defender_count:
                under += 1
    return over, under


def _extract_position_features(board, move, side):
    f = {}
    legal = list(board.legal_moves)
    f['num_legal_moves'] = len(legal)
    f['in_check'] = int(board.is_check())

    chk_moves = chk_pieces = cap_moves = cap_pieces = 0
    win_caps = safe_caps = 0
    chk_set, cap_set = set(), set()
    for lm in legal:
        is_cap = board.is_capture(lm)
        board.push(lm)
        gives_check = board.is_check()
        board.pop()
        if gives_check:
            chk_moves += 1
            chk_set.add(lm.from_square)
        if is_cap:
            cap_moves += 1
            cap_set.add(lm.from_square)
            att = board.piece_at(lm.from_square)
            vic = board.piece_at(lm.to_square)
            if att and vic:
                av, vv = PIECE_VALUES.get(att.piece_type, 0), PIECE_VALUES.get(vic.piece_type, 0)
                if vv > av:
                    win_caps += 1
                if av <= vv:
                    safe_caps += 1
    f['num_checking_moves'] = chk_moves
    f['num_checking_pieces'] = len(chk_set)
    f['num_capturing_moves'] = cap_moves
    f['num_capturing_pieces'] = len(cap_set)
    f['captures_winning_material'] = win_caps
    f['materially_safe_captures'] = safe_caps

    if move is not None:
        moving_piece = board.piece_at(move.from_square)
        board.push(move)
        f['move_is_check'] = int(board.is_check())
        board.pop()
        f['moving_piece_mobility'] = sum(1 for lm in legal if lm.from_square == move.from_square)
        captured_piece = board.piece_at(move.to_square)
        is_en_passant = board.is_capture(move) and captured_piece is None
        if captured_piece:
            f['captures_undefended'] = int(not board.attackers(captured_piece.color, move.to_square))
            f['material_gain'] = PIECE_VALUES.get(captured_piece.piece_type, 0)
        elif is_en_passant:
            f['captures_undefended'] = 0
            f['material_gain'] = PIECE_VALUES[chess.PAWN]
        else:
            f['captures_undefended'] = 0
            f['material_gain'] = 0
        f['piece_type'] = moving_piece.piece_type if moving_piece else 0
        f['from_col'] = chess.square_file(move.from_square)
        f['from_row'] = chess.square_rank(move.from_square)
        f['to_col'] = chess.square_file(move.to_square)
        f['to_row'] = chess.square_rank(move.to_square)
    else:
        for k in ('move_is_check', 'moving_piece_mobility', 'captures_undefended',
                  'material_gain', 'piece_type', 'from_col', 'from_row', 'to_col', 'to_row'):
            f[k] = 0

    f['side_material'] = _count_material(board, side)
    f['opp_material'] = _count_material(board, not side)
    f['material_diff'] = f['side_material'] - f['opp_material']
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        f[f'mobility_pt{pt}'] = _piece_mobility(board, pt, side)
    uc, uv = _undefended_pieces(board, side)
    f['undefended_count'] = uc
    f['undefended_value'] = uv
    ouc, ouv = _undefended_pieces(board, not side)
    f['opp_undefended_count'] = ouc
    f['opp_undefended_value'] = ouv
    f['attackers_near_own_king'] = _attackers_near_king(board, side, not side)
    f['defenders_near_own_king'] = _attackers_near_king(board, side, side)
    f['attackers_near_opp_king'] = _attackers_near_king(board, not side, side)
    f['defenders_near_opp_king'] = _attackers_near_king(board, not side, not side)

    f['pawn_islands'] = _pawn_islands(board, side)
    f['opp_pawn_islands'] = _pawn_islands(board, not side)
    f['doubled_pawns'] = _doubled_pawns(board, side)
    f['isolated_pawns'] = _isolated_pawns(board, side)
    f['passed_pawns'] = _passed_pawns(board, side)
    f['opp_doubled_pawns'] = _doubled_pawns(board, not side)
    f['opp_isolated_pawns'] = _isolated_pawns(board, not side)
    f['opp_passed_pawns'] = _passed_pawns(board, not side)

    ov, un = _over_under_defended(board, side)
    f['over_defended'] = ov
    f['under_defended'] = un
    oov, oun = _over_under_defended(board, not side)
    f['opp_over_defended'] = oov
    f['opp_under_defended'] = oun
    f['castle_K_white'] = int(board.has_kingside_castling_rights(chess.WHITE))
    f['castle_Q_white'] = int(board.has_queenside_castling_rights(chess.WHITE))
    f['castle_K_black'] = int(board.has_kingside_castling_rights(chess.BLACK))
    f['castle_Q_black'] = int(board.has_queenside_castling_rights(chess.BLACK))
    f['has_en_passant'] = int(board.has_legal_en_passant())
    return f


def _extract_tactical_features(board, move, prev_move, prev_board):
    if move is None:
        return {'accepted_sacrifice': 0, 'interposition_defence': 0, 'recapture': 0}
    f = {}
    previous_moved_piece = board.piece_at(prev_move.to_square) if prev_move is not None else None
    previous_captured_piece = prev_board.piece_at(prev_move.to_square) if prev_move is not None and prev_board is not None else None
    if (
        board.is_capture(move)
        and prev_move is not None
        and prev_move.to_square == move.to_square
        and previous_moved_piece is not None
        and previous_captured_piece is not None
        and PIECE_VALUES.get(previous_moved_piece.piece_type, 0) > PIECE_VALUES.get(previous_captured_piece.piece_type, 0)
    ):
        f['accepted_sacrifice'] = 1
    else:
        f['accepted_sacrifice'] = 0
    moving_piece = board.piece_at(move.from_square)
    if board.is_check() and moving_piece and moving_piece.piece_type != chess.KING and not board.is_capture(move):
        f['interposition_defence'] = 1
    else:
        f['interposition_defence'] = 0
    if prev_move and prev_board and prev_board.is_capture(prev_move) and \
       board.is_capture(move) and move.to_square == prev_move.to_square:
        f['recapture'] = 1
    else:
        f['recapture'] = 0
    return f


def build_advanced_features(df, data_file_name, max_half_moves=10):
    cache_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'{data_file_name}_advanced_features.csv')
    cache_path = os.path.normpath(cache_path)
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path).values.astype(np.float32)

    dummy_board = chess.Board()
    dummy_move = list(dummy_board.legal_moves)[0]
    sample_pos = _extract_position_features(dummy_board, dummy_move, chess.WHITE)
    sample_tac = _extract_tactical_features(dummy_board, dummy_move, None, None)
    feat_keys = list(sample_pos.keys()) + list(sample_tac.keys())
    n_per_move = len(feat_keys)

    col_names = []
    for side_prefix in ('p', 'o'):
        for i in range(5):
            for k in feat_keys:
                col_names.append(f"{side_prefix}{i}_{k}")
    n_cols = len(col_names)

    result = np.zeros((len(df), n_cols), dtype=np.float32)

    for row_idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Move features")):
        board = chess.Board(row['FEN'])
        moves_uci = str(row['Moves']).split()
        player_color = not board.turn

        prev_move, prev_board = None, None
        p_count, o_count = 0, 0

        for move_uci in moves_uci[:max_half_moves]:
            move = chess.Move.from_uci(move_uci)

            is_player = board.turn == player_color
            if is_player:
                if p_count >= 5:
                    prev_board = board.copy()
                    prev_move = move
                    board.push(move)
                    continue
                prefix_idx = p_count
                base_offset = prefix_idx * n_per_move
                p_count += 1
            else:
                if o_count >= 5:
                    prev_board = board.copy()
                    prev_move = move
                    board.push(move)
                    continue
                prefix_idx = o_count
                base_offset = 5 * n_per_move + prefix_idx * n_per_move
                o_count += 1

            pos = _extract_position_features(board, move, board.turn)
            tac = _extract_tactical_features(board, move, prev_move, prev_board)

            vals = [pos[k] for k in sample_pos.keys()] + [tac[k] for k in sample_tac.keys()]
            result[row_idx, base_offset:base_offset + n_per_move] = vals

            prev_board = board.copy()
            prev_move = move
            board.push(move)

    pd.DataFrame(result, columns=col_names).to_csv(cache_path, index=False)
    return result


def _chebyshev(sq1, sq2):
    return max(
        abs(chess.square_file(sq1) - chess.square_file(sq2)),
        abs(chess.square_rank(sq1) - chess.square_rank(sq2)),
    )


def _piece_participation_stats(fen, moves_str):
    board = chess.Board(fen)
    moves_uci = str(moves_str).split()
    player_color = not board.turn

    player_from_squares = []
    player_piece_types = set()
    player_move_distances = []

    for move_uci in moves_uci:
        try:
            move = chess.Move.from_uci(move_uci)
        except Exception:
            break
        if board.turn == player_color:
            piece = board.piece_at(move.from_square)
            if piece:
                player_piece_types.add(piece.piece_type)
                player_from_squares.append(move.from_square)
                player_move_distances.append(_chebyshev(move.from_square, move.to_square))
        board.push(move)

    num_pieces = len(set(player_from_squares))
    num_piece_types = len(player_piece_types)
    max_piece_value = max((PIECE_VALUES.get(pt, 0) for pt in player_piece_types), default=0)
    uses_queen = int(chess.QUEEN in player_piece_types)
    uses_rook = int(chess.ROOK in player_piece_types)
    uses_minor = int(chess.KNIGHT in player_piece_types or chess.BISHOP in player_piece_types)
    uses_pawn = int(chess.PAWN in player_piece_types)
    avg_move_dist = float(np.mean(player_move_distances)) if player_move_distances else 0.0

    if len(player_from_squares) >= 2:
        origin_distances = [
            _chebyshev(player_from_squares[i], player_from_squares[j])
            for i in range(len(player_from_squares))
            for j in range(i + 1, len(player_from_squares))
        ]
        spatial_spread = float(np.mean(origin_distances))
    else:
        spatial_spread = 0.0

    return {
        'num_participating_pieces': num_pieces,
        'num_piece_types': num_piece_types,
        'max_piece_value': max_piece_value,
        'uses_queen': uses_queen,
        'uses_rook': uses_rook,
        'uses_minor': uses_minor,
        'uses_pawn': uses_pawn,
        'avg_move_dist': avg_move_dist,
        'spatial_spread': spatial_spread,
    }


def extract_board_stats(fen):
    board = chess.Board(fen)
    return {
        'white_pieces': int(board.occupied_co[chess.WHITE]).bit_count(),
        'black_pieces': int(board.occupied_co[chess.BLACK]).bit_count(),
        'material_balance': sum(len(board.pieces(piece, chess.WHITE)) * val
                                for piece, val in zip([1, 2, 3, 4, 5, 6], [1, 3, 3, 5, 9, 0])) -
                            sum(len(board.pieces(piece, chess.BLACK)) * val
                                for piece, val in zip([1, 2, 3, 4, 5, 6], [1, 3, 3, 5, 9, 0]))
    }


def build_features(df, save_csv_path=None):
    if save_csv_path is None:
        save_csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'filtered_struct_features.csv')
        save_csv_path = os.path.normpath(save_csv_path)
    print(f"Struct feat path: {save_csv_path}")
    if os.path.exists(save_csv_path):
        cached = pd.read_csv(save_csv_path)
        print("read cached features")
        if len(cached) == len(df):
            print("returned cached features")
            return cached.values.astype(np.float32)
    tqdm.pandas(desc="Solution length")
    length = df['Moves'].progress_apply(lambda x: len(str(x).split())).values

    feats = pd.DataFrame(index=df.index)
    feats['SolutionLength'] = length
    tqdm.pandas(desc="Side to move")
    feats['IsWhiteToMove'] = df['FEN'].progress_apply(lambda x: 1 if x.split()[1] == 'w' else 0)
    tqdm.pandas(desc="Board stats")
    stats = df['FEN'].progress_apply(extract_board_stats).apply(pd.Series)
    feats = pd.concat([feats, stats], axis=1)
    prob_cols = [c for c in df.columns if 'success_prob_blitz' in c or 'success_prob_rapid' in c]
    feats = pd.concat([feats, df[prob_cols]], axis=1)
    tqdm.pandas(desc="Participation stats")
    participation = df.progress_apply(lambda r: _piece_participation_stats(r['FEN'], r['Moves']), axis=1).apply(pd.Series)
    feats = pd.concat([feats, participation], axis=1)

    if save_csv_path is not None:
        feats.to_csv(save_csv_path, index=False)
    return feats.values.astype(np.float32)


def build_success_prob_features(df):
    prob_cols = sorted([c for c in df.columns if 'success_prob_' in c])
    if not prob_cols:
        return np.zeros((len(df), 0), dtype=np.float32)

    probs = df[prob_cols].fillna(0).values.astype(np.float32)

    prob_mean = probs.mean(axis=1)
    prob_std = probs.std(axis=1)
    prob_min = probs.min(axis=1)
    prob_max = probs.max(axis=1)
    prob_range = prob_max - prob_min

    from scipy.stats import skew
    prob_skew = np.array([skew(row) for row in probs]).astype(np.float32)

    pairwise_diffs = np.diff(probs, axis=1)
    second_deriv = np.diff(probs, n=2, axis=1)
    inflection_idx = np.argmin(second_deriv, axis=1).astype(np.float32)
    max_pairwise_drop = pairwise_diffs.min(axis=1)

    derived = np.column_stack([
        prob_mean, prob_std, prob_min, prob_max, prob_range,
        prob_skew, inflection_idx, max_pairwise_drop,
    ]).astype(np.float32)
    return derived


def encode_themes(df, themes_csv_path="./data/p200k_themes.csv"):
    themes_df = pd.read_csv(themes_csv_path, usecols=["PuzzleId", "Themes"])
    merged = df[["PuzzleId"]].merge(themes_df, on="PuzzleId", how="left")
    merged["Themes"] = merged["Themes"].fillna("")
    themes_list = merged["Themes"].apply(lambda x: x.split() if x else [])
    mlb = MultiLabelBinarizer()
    themes_encoded = mlb.fit_transform(themes_list)
    return themes_encoded.astype(np.float32)
