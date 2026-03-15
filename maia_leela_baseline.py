import pandas as pd
import numpy as np
import os
import chess
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
import mlflow
from tqdm import tqdm
from maia2 import model, inference

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


def load_data(csv_path, embeddings_path=None, num_rows=None):
    df = pd.read_csv(csv_path)
    if num_rows:
        df = df.head(num_rows)
        
    if embeddings_path and os.path.exists(embeddings_path):
        maia_embeddings = np.load(embeddings_path)
        if len(maia_embeddings) != len(df):
            maia_embeddings = maia_embeddings[:len(df)]
    else:
        maia_embeddings = process_puzzle_sequences(df)
        
    return df, maia_embeddings

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
    
    return feats.values.astype(np.float32)

def encode_themes(df):
    themes_list = df['Themes'].apply(lambda x: x.split())
    mlb = MultiLabelBinarizer()
    themes_encoded = mlb.fit_transform(themes_list)
    return themes_encoded.astype(np.float32)


class ChessPuzzleDataset(Dataset):
    def __init__(self, X_struct, X_themes, X_maia_seq, ratings):
        self.X_struct = torch.tensor(X_struct, dtype=torch.float32)
        self.X_themes = torch.tensor(X_themes, dtype=torch.float32)
        self.X_seq = torch.tensor(X_maia_seq, dtype=torch.float32)
        self.ratings = torch.tensor(ratings, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return (self.X_struct[idx], 
                self.X_themes[idx], 
                self.X_seq[idx], 
                self.ratings[idx])


class PuzzleRatingMLP(nn.Module):
    def __init__(self, 
                 struct_in_dim, 
                 themes_in_dim, 
                 seq_embed_dim, 
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
        
        self.rnn = nn.RNN(
            input_size=seq_embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        combined_dim = (hidden_dim // 2) + (hidden_dim // 2) + hidden_dim
        
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, struct_in, themes_in, seq_in):
        struct_feats = self.struct_mlp(struct_in)
        themes_feats = self.themes_mlp(themes_in)
        
        rnn_out, rnn_hidden = self.rnn(seq_in)
        seq_feats = rnn_hidden[-1]
        
        combined = torch.cat([struct_feats, themes_feats, seq_feats], dim=1)
        
        rating_pred = self.predictor(combined)
        return rating_pred


def train_loop(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu', print_freq=1):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_mse = float('inf')
    best_val_rmse = float('inf')
    early_stop_patience = 10
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            struct, themes, seq, ratings = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            preds = model(struct, themes, seq)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(ratings)
            
        train_mse = train_loss / len(train_loader.dataset)
        train_rmse = np.sqrt(train_mse)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                struct, themes, seq, ratings = [b.to(device) for b in batch]
                preds = model(struct, themes, seq)
                loss = criterion(preds, ratings)
                val_loss += loss.item() * len(ratings)
                
        val_mse = val_loss / len(val_loader.dataset)
        val_rmse = np.sqrt(val_mse)
        
        mlflow.log_metric("train_mse", train_mse, step=epoch)
        mlflow.log_metric("val_mse", val_mse, step=epoch)
        mlflow.log_metric("train_rmse", train_rmse, step=epoch)
        mlflow.log_metric("val_rmse", val_rmse, step=epoch)
        
        if (epoch + 1) % print_freq == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_mse:.2f} (RMSE: {train_rmse:.2f}) | Val MSE: {val_mse:.2f} (RMSE: {val_rmse:.2f})")
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    model.load_state_dict(best_model_state)
    return model, best_val_mse, best_val_rmse


if __name__ == "__main__":
    
    csv_path = "./data/p200k.csv"
    embeddings_path = None
    # embeddings_path = "../features/maia2_sequence_embeddings.npy"
    
    mlflow.set_experiment("Chess_Puzzle_Rating_Prediction")
    
    df, maia_seq = load_data(csv_path, embeddings_path, num_rows=1000)
    X_struct = build_features(df)
    X_themes = encode_themes(df)
    y = df['Rating'].values
    
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    
    train_dataset = ChessPuzzleDataset(X_struct[train_idx], X_themes[train_idx], maia_seq[train_idx], y[train_idx])
    val_dataset = ChessPuzzleDataset(X_struct[val_idx], X_themes[val_idx], maia_seq[val_idx], y[val_idx])
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = PuzzleRatingMLP(
        struct_in_dim=X_struct.shape[1],
        themes_in_dim=X_themes.shape[1],
        seq_embed_dim=maia_seq.shape[2]
    ).to(device)

    epochs = 50000
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "PyTorch_MLP_RNN")
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("epochs", 50000)
        mlflow.log_param("features", "struct_themes_maiaSeq")
        mlflow.log_param("rnn_hidden_dim", 256)
        
        best_model, final_mse, final_rmse = train_loop(model, train_loader, val_loader, epochs=50000, device=device, print_freq=100)
        
        print(f"Training complete. Validation MSE: {final_mse:.2f} | Validation RMSE: {final_rmse:.2f}")
        mlflow.pytorch.log_model(best_model, "maia_leela_baseline_model")
        
        out_dir = f"./results/p200k"
        os.makedirs(out_dir, exist_ok=True)
        result = pd.DataFrame([{'Validation_MSE': final_mse, 'Validation_RMSE': final_rmse}])
        result.to_csv(f"{out_dir}/lenInStructFeats.csv", index=False)
