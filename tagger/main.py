import os
import chess
import chess.pgn
import chess.engine
import pandas as pd
from tqdm import tqdm

from model import Puzzle
from themes import cook

def get_puzzle_data(puzzle_id: str, fen: str, moves_list: str, engine: chess.engine.SimpleEngine):
    try:
        board = chess.Board(fen)
        
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
    OUTPUT_CSV = "../../filtered_themes.csv"
    STOCKFISH_PATH = "../stockfish/stockfish"
    
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    df = pd.read_csv(INPUT_CSV)
    
    required_cols = ['PuzzleId', 'FEN', 'Moves']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain '{col}' column.")
            
    tqdm.pandas(desc="Tagging Puzzles")
    
    results = df.progress_apply(
        lambda row: get_puzzle_data(
            str(row['PuzzleId']), 
            str(row['FEN']), 
            str(row['Moves']), 
            engine
        ), 
        axis=1, 
        result_type='expand'
    )
    
    df['cp'] = results[0]
    df['Themes'] = results[1]
    
    df.to_csv(OUTPUT_CSV, index=False)
    
    engine.quit()

if __name__ == "__main__":
    main()
