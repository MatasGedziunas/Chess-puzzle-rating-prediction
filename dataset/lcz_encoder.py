import re
import sys
import os
import chess
import numpy as np

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from policy_index import policy_index

_MOVE_MAP = dict(zip(policy_index, range(len(policy_index))))
_MOVE_RE = re.compile(r"^([a-h])(\d)([a-h])(\d)(.*)$")


def mirror_uci(uci):
    m = _MOVE_RE.match(uci)
    return f"{m.group(1)}{9 - int(m.group(2))}{m.group(3)}{9 - int(m.group(4))}{m.group(5)}"


def board_to_planes(board):
    b = board.mirror() if not board.turn else board

    planes = np.zeros((13, 8, 8), dtype=np.float32)
    piece_map = {
        (chess.PAWN,   chess.WHITE): 0,  (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,  (chess.ROOK,   chess.WHITE): 3,
        (chess.QUEEN,  chess.WHITE): 4,  (chess.KING,   chess.WHITE): 5,
        (chess.PAWN,   chess.BLACK): 6,  (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,  (chess.ROOK,   chess.BLACK): 9,
        (chess.QUEEN,  chess.BLACK): 10, (chess.KING,   chess.BLACK): 11,
    }
    for sq in chess.SQUARES:
        piece = b.piece_at(sq)
        if piece:
            row, col = sq // 8, sq % 8
            planes[piece_map[(piece.piece_type, piece.color)], row, col] = 1.0

    repeated = np.tile(planes, (8, 1, 1))

    castling = np.array([
        float(bool(b.castling_rights & chess.BB_H1)),
        float(bool(b.castling_rights & chess.BB_A1)),
        float(bool(b.castling_rights & chess.BB_H8)),
        float(bool(b.castling_rights & chess.BB_A8)),
        float(not board.turn),
        0.0,
        0.0,
        1.0,
    ], dtype=np.float32)
    scalar_planes = castling[:, np.newaxis, np.newaxis] * np.ones((8, 8, 8), dtype=np.float32)

    return np.concatenate([repeated, scalar_planes], axis=0)


def uci_to_policy_idx(board, uci):
    fixed = mirror_uci(uci) if not board.turn else uci

    mirrored_board = board.mirror() if not board.turn else board
    try:
        move = chess.Move.from_uci(fixed)
        if fixed == "e1g1" and mirrored_board.is_kingside_castling(move):
            fixed = "e1h1"
        elif fixed == "e1c1" and mirrored_board.is_queenside_castling(move):
            fixed = "e1a1"
    except ValueError:
        pass
    if fixed.endswith("n"):
        fixed = fixed[:-1]

    return _MOVE_MAP.get(fixed)
