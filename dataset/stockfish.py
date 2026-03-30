import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess
import pandas as pd
from tqdm import tqdm


def _fen_after_first_move(fen, moves_str):
    board = chess.Board(fen)
    board.push_uci(moves_str.strip().split()[0])
    return board.fen()


def get_stockfish_features(row):
    STOCKFISH_PATH = "./stockfish/stockfish"
    puzzle_id, fen, moves = row['PuzzleId'], row['FEN'], row['Moves']

    eval_fen = _fen_after_first_move(fen, moves)

    process = subprocess.Popen(
        [STOCKFISH_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    commands = f"position fen {eval_fen}\neval\nquit\n"
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

        process2.stdin.write(f"position fen {eval_fen}\ngo depth 10\n")
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
        done_df = existing_df[existing_df['SF_Final_Eval'].notna()]
        already_processed = set(done_df['PuzzleId'].tolist())
        existing_results = done_df.to_dict('records')

    tasks = df[~df['PuzzleId'].isin(already_processed)][['PuzzleId', 'FEN', 'Moves']].to_dict('records')
    if len(tasks) == 0:
        return

    new_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_stockfish_features, row): row for row in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Puzzles"):
            try:
                res = future.result()
                new_results.append(res)
            except Exception:
                pass

    all_results = existing_results + new_results
    pd.DataFrame(all_results).to_csv(output_csv, index=False)
