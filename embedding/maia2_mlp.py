import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Maia2MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256), embed_dim=128):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, embed_dim))
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        emb = self.encoder(x)
        return emb, self.head(emb).squeeze(1)


def train_mlp_embedder(
    X_train, y_train,
    X_val, y_val,
    embed_dim=128,
    hidden_dims=(512, 256),
    epochs=50,
    batch_size=2048,
    lr=1e-3,
    save_path=None,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    model = Maia2MLP(X_train.shape[1], hidden_dims=hidden_dims, embed_dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None

    for epoch in tqdm(range(epochs), desc="MLP training"):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            _, val_pred = model(X_v.to(device))
            val_mse = criterion(val_pred, y_v.to(device)).item()

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    if save_path:
        torch.save(best_state, save_path)
        print(f"MLP saved -> {save_path}  best_val_mse={best_val_mse:.2f}")

    return model, best_val_mse


def extract_embeddings(model, X, batch_size=4096, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size)
    parts = []
    with torch.no_grad():
        for (xb,) in loader:
            emb, _ = model(xb.to(device))
            parts.append(emb.cpu().numpy())
    return np.concatenate(parts, axis=0)


def load_or_train_mlp_embedder(
    X_train, y_train,
    X_val, y_val,
    X_full,
    input_dim,
    cache_dir="./data",
    data_file_name="filtered",
    embed_dim=128,
    hidden_dims=(512, 256),
    epochs=50,
    batch_size=2048,
    lr=1e-3,
):
    model_path = os.path.join(cache_dir, f"{data_file_name}_maia2_mlp.pt")
    emb_path = os.path.join(cache_dir, f"{data_file_name}_maia2_mlp_embeddings.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Maia2MLP(input_dim, hidden_dims=hidden_dims, embed_dim=embed_dim).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded MLP from {model_path}")
    else:
        model, val_mse = train_mlp_embedder(
            X_train, y_train, X_val, y_val,
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_path=model_path,
            device=device,
        )
        print(f"MLP val MSE: {val_mse:.2f}")

    if os.path.exists(emb_path):
        embeddings = np.load(emb_path)
        print(f"Loaded MLP embeddings from {emb_path}  shape={embeddings.shape}")
    else:
        embeddings = extract_embeddings(model, X_full, device=device)
        np.save(emb_path, embeddings)
        print(f"Saved MLP embeddings -> {emb_path}  shape={embeddings.shape}")

    return embeddings
