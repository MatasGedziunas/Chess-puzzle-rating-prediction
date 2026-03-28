import numpy as np
import torch
from torch.utils.data import Dataset


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
