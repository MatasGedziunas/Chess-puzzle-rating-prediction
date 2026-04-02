import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow

from dataset.loaders import load_data
from dataset.board_features import build_features, encode_themes
from dataset.torch_dataset import ChessPuzzleDataset
from MlpModel import PuzzleRatingMLP


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
            struct, themes, seq, lengths, ratings = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            preds = model(struct, themes, seq, lengths)
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
                struct, themes, seq, lengths, ratings = [b.to(device) for b in batch]
                preds = model(struct, themes, seq, lengths)
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
    move_lengths = df['Moves'].apply(lambda x: len(str(x).split())).values
    X_themes = encode_themes(df)
    y = df['Rating'].values
    
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    
    train_dataset = ChessPuzzleDataset(X_struct[train_idx], X_themes[train_idx], maia_seq[train_idx], move_lengths[train_idx], y[train_idx])
    val_dataset = ChessPuzzleDataset(X_struct[val_idx], X_themes[val_idx], maia_seq[val_idx], move_lengths[val_idx], y[val_idx])
    
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
        mlflow.log_param("features", "struct_themes_maiaSeq_len")
        mlflow.log_param("rnn_hidden_dim", 256)
        
        best_model, final_mse, final_rmse = train_loop(model, train_loader, val_loader, epochs=50000, device=device, print_freq=100)
        
        print(f"Training complete. Validation MSE: {final_mse:.2f} | Validation RMSE: {final_rmse:.2f}")
        mlflow.pytorch.log_model(best_model, "maia_leela_baseline_model")
        
        out_dir = f"./results/p200k"
        os.makedirs(out_dir, exist_ok=True)
        result = pd.DataFrame([{'Validation_MSE': final_mse, 'Validation_RMSE': final_rmse}])
        result.to_csv(f"{out_dir}/maia_leela_baseline_results.csv", index=False)
