from chessPuzzle.dataset.loaders import load_data, load_stockfish_features
from chessPuzzle.dataset.board_features import (
    build_features,
    encode_themes,
    build_advanced_features,
    extract_board_stats,
    PIECE_VALUES,
)
from chessPuzzle.dataset.maia2_embeddings import FeatureExtractor, process_puzzle_sequences
from chessPuzzle.dataset.stockfish import get_stockfish_features, process_all_puzzles
from chessPuzzle.dataset.torch_dataset import ChessPuzzleDataset


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.latent_vector = None
        self.hook = self.model.last_ln.register_forward_hook(self.save_output)

    def save_output(self, module, input, output):
        self.latent_vector = output

    def __call__(self, boards, elo_self, elo_oppo):
        self.model(boards, elos_self=elo_self, elos_oppo=elo_oppo)
        return self.latent_vector


def get_stockfish_features(row):
    STOCKFISH_PATH = "./stockfish" 
    puzzle_id, fen = row['PuzzleId'], row['FEN']
    
    process = subprocess.Popen(
        [STOCKFISH_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    commands = f"position fen {fen}\neval\nquit\n"
    output, _ = process.communicate(commands)
    
    metrics = {
        "PuzzleId": puzzle_id,
        "SF_Material": None,
        "SF_Positional": None,
        "SF_Final_Eval": None
    }
    
    if "Final evaluation: none (in check)" in output:
        process2 = subprocess.Popen(
            [STOCKFISH_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process2.stdin.write(f"position fen {fen}\ngo depth 10\n")
        process2.stdin.flush()
        
        for line in process2.stdout:
            line = line.strip()
            if 'score cp' in line:
                match = re.search(r'score cp (-?\d+)', line)
                if match:
                    metrics["SF_Final_Eval"] = int(match.group(1)) / 100.0
            elif 'score mate' in line:
                match = re.search(r'score mate (-?\d+)', line)
                if match:
                    mate_in = int(match.group(1))
                    metrics["SF_Final_Eval"] = 100.0 if mate_in > 0 else -100.0
            elif line.startswith('bestmove'):
                break

        
        process2.stdin.write("quit\n")
        process2.stdin.flush()
        process2.wait()

    else:
        # Normal parsing for non-check positions
        lines = output.split('\n')
        for line in lines:
            if "<-- this bucket is used" in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 5:
                    try:
                        metrics["SF_Material"] = float(parts[2].replace(" ", ""))
                        metrics["SF_Positional"] = float(parts[3].replace(" ", ""))
                    except ValueError:
                        pass
                        
            elif "Final evaluation" in line and "(white side)" in line:
                match = re.search(r"([+-]?\d+\.\d+)", line)
                if match:
                    metrics["SF_Final_Eval"] = float(match.group(1))
                    
    return metrics

    
def process_all_puzzles(input_csv, output_csv, max_workers=8):
    df = pd.read_csv(input_csv)
    
    already_processed = set()
    existing_results = []
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        done_mask = (
            # existing_df['SF_Material'].notna() &
            # existing_df['SF_Positional'].notna() &
            existing_df['SF_Final_Eval'].notna()
        )
        done_df = existing_df[done_mask]
        already_processed = set(done_df['PuzzleId'].tolist())
        existing_results = done_df.to_dict('records')
        print(f"Loaded {len(already_processed)} already-processed puzzles from {output_csv}")
    
    tasks = df[~df['PuzzleId'].isin(already_processed)][['PuzzleId', 'FEN']].to_dict('records')
    print(f"Remaining puzzles to process: {len(tasks)}")
    
    if len(tasks) == 0:
        print("All puzzles already processed.")
        return
    
    new_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_stockfish_features, row): row for row in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Puzzles"):
            try:
                res = future.result()
                new_results.append(res)
            except Exception as e:
                print(f"Error processing puzzle: {e}")
    
    all_results = existing_results + new_results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved {len(all_results)} total results to {output_csv}")



def process_puzzle_sequences(df, elo_indices=[0, 5, 10], max_steps=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    maia2_model = model.from_pretrained(type="rapid", device=device)
    maia2_model.to(device)
    maia2_model.eval()
    
    extractor = FeatureExtractor(maia2_model)

    num_puzzles = len(df)
    num_elos = len(elo_indices)
    sequence_results = np.zeros((num_puzzles, max_steps, num_elos, 1024), dtype=np.float32)
    
    for p_idx, row in tqdm(df.iterrows(), total=num_puzzles, desc="Computing Maia2 Embeddings"):
        board = chess.Board(row['FEN'])
        moves = str(row['Moves']).split()
        
        puzzle_fens = [board.fen()]
        for move_str in moves[:max_steps-1]:
            try:
                move = chess.Move.from_uci(move_str)
                board.push(move)
                puzzle_fens.append(board.fen())
            except:
                break
        
        num_steps = len(puzzle_fens)
        batch_tensors = [inference.board_to_tensor(chess.Board(f)) for f in puzzle_fens]
        boards = torch.stack(batch_tensors).to(device)
        
        for elo_idx_in_list, skill_val in enumerate(elo_indices):
            elos = torch.full((num_steps,), skill_val, device=device).long()
            with torch.no_grad():
                latent = extractor(boards, elos, elos)
                if len(latent.shape) > 2:
                    latent = latent.mean(dim=1)  
                sequence_results[p_idx, :num_steps, elo_idx_in_list, :] = latent.cpu().numpy()
                
    sequence_results = sequence_results.reshape(num_puzzles, max_steps, -1)
    return sequence_results


def load_data(csv_path, embeddings_path=None, stockfish_path=None, num_rows=None):
    df = pd.read_csv(csv_path)
    if num_rows:
        df = df.head(num_rows)
        
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
        print(f"Saved computed embeddings to {save_path}")
        
    if stockfish_path:
        sf_df = pd.read_csv(stockfish_path)
        df = df.merge(sf_df, on='PuzzleId', how='left')
        sf_cols = ['SF_Material', 'SF_Positional', 'SF_Final_Eval']
        stockfish_features = df[sf_cols].fillna(0).values.astype(np.float32)
    else: 
        stockfish_features = None
    
    return df, maia_embeddings, stockfish_features


def extract_board_stats(fen):
    board = chess.Board(fen)
    counts = {
        'white_pieces': int(board.occupied_co[chess.WHITE]).bit_count(),
        'black_pieces': int(board.occupied_co[chess.BLACK]).bit_count(),
        'material_balance': sum([len(board.pieces(piece, chess.WHITE)) * val 
                               for piece, val in zip([1,2,3,4,5,6], [1,3,3,5,9,0])]) - \
                           sum([len(board.pieces(piece, chess.BLACK)) * val 
                               for piece, val in zip([1,2,3,4,5,6], [1,3,3,5,9,0])])
    }
    return counts


def build_features(df):
    feats = pd.DataFrame(index=df.index)
    feats['SolutionLength'] = df['Moves'].apply(lambda x: len(str(x).split()))
    feats['IsWhiteToMove'] = df['FEN'].apply(lambda x: 1 if x.split()[1] == 'w' else 0)
    stats = df['FEN'].apply(extract_board_stats).apply(pd.Series)
    feats = pd.concat([feats, stats], axis=1)
    prob_cols = [c for c in df.columns if 'success_prob_blitz' in c]
    feats = pd.concat([feats, df[prob_cols]], axis=1)
    
    length = feats.pop('SolutionLength').values
    return feats.values.astype(np.float32), length


def load_stockfish_features(stockfish_evals_path, puzzles_df):
    sf_df = pd.read_csv(stockfish_evals_path)
    sf_df = sf_df[sf_df['PuzzleId'].isin(puzzles_df['PuzzleId'])]
    merged = puzzles_df[['PuzzleId']].merge(sf_df, on='PuzzleId', how='left')
    sf_cols = ['SF_Material', 'SF_Positional', 'SF_Final_Eval']
    return merged[sf_cols].fillna(0).values.astype(np.float32)


# ── 2025 Winner Table I & II feature helpers ──────────────────────────────

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
    king_sq = board.king(king_color)
    if king_sq is None:
        return 0
    kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
    count = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            f, r = kf + df, kr + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                count += len(board.attackers(attacking_color, chess.square(f, r)))
    return count


def _pawn_islands(board, color):
    files = sorted({chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)})
    if not files:
        return 0
    return 1 + sum(1 for i in range(1, len(files)) if files[i] - files[i - 1] > 1)


def _doubled_pawns(board, color):
    fcounts = [0] * 8
    for sq in board.pieces(chess.PAWN, color):
        fcounts[chess.square_file(sq)] += 1
    return sum(c - 1 for c in fcounts if c > 1)


def _isolated_pawns(board, color):
    pf = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
    return sum(1 for sq in board.pieces(chess.PAWN, color)
               if (f := chess.square_file(sq)) and (f - 1) not in pf and (f + 1) not in pf)


def _passed_pawns(board, color):
    opp = not color
    direction = 1 if color == chess.WHITE else -1
    count = 0
    for sq in board.pieces(chess.PAWN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        is_passed = True
        for cf in (f - 1, f, f + 1):
            if 0 <= cf <= 7:
                cr = r + direction
                while 0 <= cr <= 7:
                    p = board.piece_at(chess.square(cf, cr))
                    if p and p.piece_type == chess.PAWN and p.color == opp:
                        is_passed = False
                        break
                    cr += direction
                if not is_passed:
                    break
        if is_passed:
            count += 1
    return count


def _undefended_pieces(board, color):
    cnt, val = 0, 0
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board.pieces(pt, color):
            if not board.attackers(color, sq):
                cnt += 1
                val += PIECE_VALUES[pt]
    return cnt, val


def _over_under_defended(board, color):
    opp = not color
    over, under = 0, 0
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board.pieces(pt, color):
            d = len(board.attackers(color, sq))
            a = len(board.attackers(opp, sq))
            if d > a and a > 0:
                over += 1
            elif a > d:
                under += 1
    return over, under


def _extract_position_features(board, move, side):
    """All Table I features for one position + correct move."""
    f = {}
    legal = list(board.legal_moves)
    f['num_legal_moves'] = len(legal)
    f['in_check'] = int(board.is_check())

    # ── Forcing moves (iterate legal moves once) ──
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

    # ── Correct move features ──
    if move is not None:
        mp = board.piece_at(move.from_square)
        board.push(move)
        f['move_is_check'] = int(board.is_check())
        board.pop()
        f['moving_piece_mobility'] = sum(1 for lm in legal if lm.from_square == move.from_square)
        cap = board.piece_at(move.to_square)
        # handle en-passant: no piece at to_square but still a capture
        is_ep = (board.is_capture(move) and cap is None)
        if cap:
            f['captures_undefended'] = int(not board.attackers(cap.color, move.to_square))
            f['material_gain'] = PIECE_VALUES.get(cap.piece_type, 0)
        elif is_ep:
            f['captures_undefended'] = 0
            f['material_gain'] = PIECE_VALUES[chess.PAWN]
        else:
            f['captures_undefended'] = 0
            f['material_gain'] = 0
        f['piece_type'] = mp.piece_type if mp else 0
        f['from_col'] = chess.square_file(move.from_square)
        f['from_row'] = chess.square_rank(move.from_square)
        f['to_col'] = chess.square_file(move.to_square)
        f['to_row'] = chess.square_rank(move.to_square)
    else:
        for k in ('move_is_check', 'moving_piece_mobility', 'captures_undefended',
                   'material_gain', 'piece_type', 'from_col', 'from_row', 'to_col', 'to_row'):
            f[k] = 0

    # ── Material & structure ──
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

    # ── Pawn structure ──
    f['pawn_islands'] = _pawn_islands(board, side)
    f['opp_pawn_islands'] = _pawn_islands(board, not side)
    f['doubled_pawns'] = _doubled_pawns(board, side)
    f['isolated_pawns'] = _isolated_pawns(board, side)
    f['passed_pawns'] = _passed_pawns(board, side)
    f['opp_doubled_pawns'] = _doubled_pawns(board, not side)
    f['opp_isolated_pawns'] = _isolated_pawns(board, not side)
    f['opp_passed_pawns'] = _passed_pawns(board, not side)

    # ── Other ──
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
    """Table II features for one position + correct move."""
    if move is None:
        return {'accepted_sacrifice': 0, 'interposition_defence': 0, 'recapture': 0}
    f = {}
    # Accepted sacrifice: opponent moved piece to a square where we can capture it
    if board.is_capture(move) and prev_move is not None and prev_move.to_square == move.to_square:
        f['accepted_sacrifice'] = 1
    else:
        f['accepted_sacrifice'] = 0
    # Interposition defence: we are in check; correct move blocks it (non-king, non-capture)
    mp = board.piece_at(move.from_square)
    if board.is_check() and mp and mp.piece_type != chess.KING and not board.is_capture(move):
        f['interposition_defence'] = 1
    else:
        f['interposition_defence'] = 0
    # Recapture: opponent just captured, and we recapture on the same square
    if prev_move and prev_board and prev_board.is_capture(prev_move) and \
       board.is_capture(move) and move.to_square == prev_move.to_square:
        f['recapture'] = 1
    else:
        f['recapture'] = 0
    return f


def build_advanced_features(df, max_half_moves=10):
    """
    Compute per-move features (Tables I & II, 2025 winner).
    Returns np.ndarray of shape (n_puzzles, n_features).
    Features prefixed p0_..p4_ (player) and o0_..o4_ (opponent).
    """
    # Determine feature keys from a dummy extraction
    dummy_board = chess.Board()
    dummy_move = list(dummy_board.legal_moves)[0]
    sample_pos = _extract_position_features(dummy_board, dummy_move, chess.WHITE)
    sample_tac = _extract_tactical_features(dummy_board, dummy_move, None, None)
    feat_keys = list(sample_pos.keys()) + list(sample_tac.keys())
    n_per_move = len(feat_keys)

    # Column order: p0_feat0, p0_feat1, ..., p4_featN, o0_feat0, ..., o4_featN
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
        player_color = not board.turn  # first mover is opponent

        prev_move, prev_board = None, None
        p_count, o_count = 0, 0

        for move_uci in moves_uci[:max_half_moves]:
            try:
                move = chess.Move.from_uci(move_uci)
            except Exception:
                break

            is_player = (board.turn == player_color)
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

    return result


def encode_themes(df):
    themes_list = df['Themes'].apply(lambda x: x.split())
    mlb = MultiLabelBinarizer()
    themes_encoded = mlb.fit_transform(themes_list)
    return themes_encoded.astype(np.float32)

    


class ChessPuzzleDataset(Dataset):
    def __init__(self, X_struct, X_themes, X_maia_seq, move_lengths, ratings):
        self.X_struct = torch.tensor(X_struct, dtype=torch.float32)
        self.X_themes = torch.tensor(X_themes, dtype=torch.float32)
        self.X_seq = torch.tensor(X_maia_seq, dtype=torch.float32)
        self.lengths = torch.tensor(np.clip(move_lengths, 0, 15), dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return (self.X_struct[idx], 
                self.X_themes[idx], 
                self.X_seq[idx], 
                self.lengths[idx], 
                self.ratings[idx])
