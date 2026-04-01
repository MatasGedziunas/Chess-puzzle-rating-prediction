import chess
import chess.pgn
import chess.engine
import pandas as pd
from tqdm import tqdm

from model import Puzzle
from themes import cook


def get_puzzle_data(puzzle_id: str, fen: str, moves_list: str, precomputed_cp=None, engine=None):
    try:
        board = chess.Board(fen)

        cp = precomputed_cp
        if cp is None:
            if engine is None:
                raise ValueError("Missing engine for puzzles without precomputed cp")
            info = engine.analyse(board, chess.engine.Limit(depth=12))
            score = info["score"].pov(not board.turn)
            cp = score.score(mate_score=10000)
        
        game = chess.pgn.Game()
        game.setup(board)
        node = game
        
        for move_uci in moves_list.split():
            move = chess.Move.from_uci(move_uci)
            node = node.add_variation(move)
            
        puzzle = Puzzle(id=str(puzzle_id), game=game, cp=cp)
        themes = cook(puzzle)
        
        return cp, " ".join(themes)
        
    except Exception as e:
        print(f"Error parsing puzzle {puzzle_id}: {e}")
        return None, ""

def main():
    INPUT_CSV = "../../filtered.csv"
    STOCKFISH_FEATURES_CSV = "../../filtered_sf_evals.csv"
    OUTPUT_CSV = "../../filtered_themes.csv"
    STOCKFISH_PATH = "../stockfish/stockfish"

    df = pd.read_csv(INPUT_CSV)
    stockfish_df = pd.read_csv(STOCKFISH_FEATURES_CSV, usecols=['PuzzleId', 'SF_Last_Move_CP'])
    df = df.merge(stockfish_df, on='PuzzleId', how='left')

    required_cols = ['PuzzleId', 'FEN', 'Moves']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain '{col}' column.")

    if 'SF_Last_Move_CP' not in df.columns:
        raise ValueError("Stockfish features CSV must contain 'SF_Last_Move_CP' column.")

    precomputed_cp = df['SF_Last_Move_CP']
    needs_engine = precomputed_cp.isna().any()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) if needs_engine else None

    tqdm.pandas(desc="Tagging Puzzles")

    results = df.progress_apply(
        lambda row: get_puzzle_data(
            str(row['PuzzleId']),
            str(row['FEN']),
            str(row['Moves']),
            precomputed_cp=int(row['SF_Last_Move_CP']) if pd.notna(row['SF_Last_Move_CP']) else None,
            engine=engine
        ),
        axis=1,
        result_type='expand'
    )

    df['cp'] = results[0]
    df['Themes'] = results[1]

    df.to_csv(OUTPUT_CSV, index=False)

    if engine is not None:
        engine.quit()

if __name__ == "__main__":
    main()
