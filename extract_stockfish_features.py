from dataset.stockfish import process_all_puzzles

process_all_puzzles("./data/p200k.csv", "./data/p200k_sf_evals.csv", max_workers=8)
