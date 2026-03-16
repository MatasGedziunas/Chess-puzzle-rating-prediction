import torch
import torch.nn as nn


class PuzzleRatingMLP(nn.Module):
    def __init__(self, 
                 struct_in_dim, 
                 themes_in_dim, 
                 seq_embed_dim, 
                 max_moves=16, 
                 move_embed_dim=32,
                 hidden_dim=256):
        super(PuzzleRatingMLP, self).__init__()
        
        self.struct_mlp = nn.Sequential(
            nn.Linear(struct_in_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.themes_mlp = nn.Sequential(
            nn.Linear(themes_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.length_embed = nn.Embedding(max_moves, move_embed_dim)
        
        self.rnn = nn.RNN(
            input_size=seq_embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        combined_dim = (hidden_dim // 2) + (hidden_dim // 2) + move_embed_dim + hidden_dim
        
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, struct_in, themes_in, seq_in, length_in):
        struct_feats = self.struct_mlp(struct_in)
        themes_feats = self.themes_mlp(themes_in)
        len_feats = self.length_embed(length_in)
        
        rnn_out, rnn_hidden = self.rnn(seq_in)
        seq_feats = rnn_hidden[-1]
        
        combined = torch.cat([struct_feats, themes_feats, seq_feats, len_feats], dim=1)
        
        rating_pred = self.predictor(combined)
        return rating_pred
