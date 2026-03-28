import chess
import numpy as np
import torch
from maia2 import model, inference
from tqdm import tqdm


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
        for move_str in moves[:max_steps - 1]:
            try:
                move = chess.Move.from_uci(move_str)
                board.push(move)
                puzzle_fens.append(board.fen())
            except Exception:
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
