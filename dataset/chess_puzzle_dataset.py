import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from .loaders import load_data, load_maia2_features
from .board_features import build_features, encode_themes, build_advanced_features, build_success_prob_features


class ChessPuzzleDataset:
    def __init__(
        self,
        csv_path,
        data_dir="./data",
        stockfish_path=None,
        themes_csv_path=None,
        use_maia2=True,
        use_maia2_mlp=False,
        filter_rating_deviation=True,
        max_rows=None,
    ):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.stockfish_path = stockfish_path
        self.themes_csv_path = themes_csv_path
        self.use_maia2 = use_maia2
        self.use_maia2_mlp = use_maia2_mlp
        self.filter_rating_deviation = filter_rating_deviation
        self.max_rows = max_rows
        self.data_file_name = os.path.splitext(os.path.basename(csv_path))[0]

    def _cache_path(self):
        suffix = ""
        if not self.use_maia2:
            suffix += "_nomaia2"
        if self.use_maia2_mlp:
            suffix += "_mlp"
        if self.max_rows:
            suffix += f"_n{self.max_rows}"
        if self.filter_rating_deviation:
            suffix += "_frd"
        return os.path.join(self.data_dir, f"{self.data_file_name}_features{suffix}.npz")

    def _build_features(self):
        print(f"Building features for {self.csv_path}")
        df, stockfish_features = load_data(
            self.csv_path,
            stockfish_path=self.stockfish_path,
            num_rows=self.max_rows,
        )

        if self.filter_rating_deviation and "RatingDeviation" in df.columns:
            df = df[df["RatingDeviation"] <= 90].reset_index(drop=True)
            if stockfish_features is not None:
                stockfish_features = stockfish_features[df.index]

        y = df["Rating"].values.astype(np.float32)

        X_struct = build_features(df)
        X_themes = encode_themes(df, themes_csv_path=self.themes_csv_path)
        advanced_features = build_advanced_features(df, self.data_file_name)
        success_prob_features = build_success_prob_features(df)

        parts = [X_struct, X_themes, advanced_features, success_prob_features]
        feature_log = {
            "struct": X_struct.shape[1],
            "themes": X_themes.shape[1],
            "advanced": advanced_features.shape[1],
            "success_prob": success_prob_features.shape[1],
        }

        if stockfish_features is not None:
            parts.append(stockfish_features)
            feature_log["stockfish"] = stockfish_features.shape[1]

        if self.use_maia2:
            maia2_features = load_maia2_features(self.data_file_name, self.data_dir)
            if maia2_features is not None:
                parts.append(maia2_features[:len(df)])
                feature_log["maia2"] = maia2_features.shape[1]

        X = np.concatenate(parts, axis=1).astype(np.float32)

        print(f"Feature dimensions: {feature_log}")
        print(f"Total: {X.shape[1]} features, {len(X)} rows")

        return X, y, df

    def load_maia2_only(self):
        return load_maia2_features(self.data_file_name, self.data_dir)

    def _append_mlp_embeddings(self, X, y, train_idx, val_idx):
        from embedding.maia2_mlp import load_or_train_mlp_embedder
        maia2_features = self.load_maia2_only()
        if maia2_features is None:
            return X
        maia2_embeddings = load_or_train_mlp_embedder(
            maia2_features[train_idx], y[train_idx],
            maia2_features[val_idx], y[val_idx],
            maia2_features,
            input_dim=maia2_features.shape[1],
            cache_dir=self.data_dir,
            data_file_name=self.data_file_name,
        )
        return np.concatenate([X, maia2_embeddings], axis=1)

    def load(self, train_idx=None, val_idx=None):
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            cached = np.load(cache_path, allow_pickle=True)
            X = cached["X"]
            y = cached["y"]
            df = pd.DataFrame(cached["df_records"].item())
            print(f"Loaded X={X.shape}, y={y.shape}")
            return X, y, df

        X, y, df = self._build_features()

        if self.use_maia2_mlp:
            if train_idx is None or val_idx is None:
                n = len(y)
                indices = np.arange(n)
                train_idx_, test_idx_ = train_test_split(indices, test_size=0.1, random_state=42)
                train_idx_, val_idx_ = train_test_split(train_idx_, test_size=1.0 / 9.0, random_state=42)
                train_idx, val_idx = train_idx_, val_idx_
            X = self._append_mlp_embeddings(X, y, train_idx, val_idx)

        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            df_records=np.array(df.to_dict("records"), dtype=object),
        )
        print(f"Saved to {cache_path}")
        return X, y, df

    def _build_features(self):
