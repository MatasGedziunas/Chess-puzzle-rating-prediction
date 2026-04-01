from dataset.stockfish import process_all_puzzles

process_all_puzzles("../filtered.csv", "../filtered_sf_evals.csv", max_workers=32)
