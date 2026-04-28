import os
import json
import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split

from .loaders import load_data, load_maia1_features, load_maia2_features
from .board_features import build_features, encode_themes, build_advanced_features


class ChessPuzzleDataset:
    def __init__(
        self,
        csv_path,
        data_dir="./data",
        stockfish_path=None,
        themes_csv_path=None,
        use_maia1=True,
        use_maia2=True,
        use_maia2_mlp=False,
        filter_rating_deviation=True,
        max_rows=None,
        blocks=None,
    ):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.stockfish_path = stockfish_path
        self.themes_csv_path = themes_csv_path
        self.use_maia1 = use_maia1
        self.use_maia2 = use_maia2
        self.use_maia2_mlp = use_maia2_mlp
        self.filter_rating_deviation = filter_rating_deviation
        self.max_rows = max_rows
        self.blocks = blocks
        self.data_file_name = os.path.splitext(os.path.basename(csv_path))[0]

    def _filter_mask(self, df):
        if not self.filter_rating_deviation:
            return np.ones(len(df), dtype=bool)
        if "RatingDeviation" not in df.columns or "NbPlays" not in df.columns:
            return np.ones(len(df), dtype=bool)
        return ((df["RatingDeviation"] <= 90) & (df["NbPlays"] > 150)).to_numpy()

    def _cache_path(self):
        suffix = ""
        if self.max_rows:
            suffix += f"_n{self.max_rows}"
        if self.filter_rating_deviation:
            suffix += "_frd"
        return os.path.join(self.data_dir, f"{self.data_file_name}_features{suffix}.npz")

    def _requested_blocks(self):
        if self.blocks is not None:
            return list(self.blocks)
        blocks = ["struct", "themes", "advanced"]
        if self.stockfish_path:
            blocks.append("stockfish")
        if self.use_maia2:
            blocks.append("maia2")
        if self.use_maia1:
            blocks.append("maia1")
        if self.use_maia2_mlp:
            blocks.append("maia2_mlp")
        return blocks

    def _compute_block(self, name, df, y, stockfish_features, train_idx, val_idx):
        if name == "struct":
            return build_features(df)
        if name == "themes":
            return encode_themes(df, themes_csv_path=self.themes_csv_path)
        if name == "advanced":
            return build_advanced_features(df, self.data_file_name)
        if name == "stockfish":
            return stockfish_features
        if name == "maia2":
            return load_maia2_features(self.data_file_name, self.data_dir)
        if name == "maia1":
            return load_maia1_features(self.data_file_name, self.data_dir)
        if name == "maia2_mlp":
            return self._compute_mlp_block(y, train_idx, val_idx)
        return None

    def _compute_mlp_block(self, y, train_idx, val_idx):
        from embedding.maia2_mlp import load_or_train_mlp_embedder
        maia2_features = load_maia2_features(self.data_file_name, self.data_dir)
        if maia2_features is None:
            return None
        if train_idx is None or val_idx is None:
            n = len(y)
            indices = np.arange(n)
            train_idx_, test_idx_ = train_test_split(indices, test_size=0.1, random_state=42)
            train_idx, val_idx = train_test_split(train_idx_, test_size=1.0 / 9.0, random_state=42)
        return load_or_train_mlp_embedder(
            maia2_features[train_idx], y[train_idx],
            maia2_features[val_idx], y[val_idx],
            maia2_features,
            input_dim=maia2_features.shape[1],
            cache_dir=self.data_dir,
            data_file_name=self.data_file_name,
        )

    def load(self, train_idx=None, val_idx=None):
        cache_path = self._cache_path()
        requested = self._requested_blocks()

        cached_blocks = {}
        manifest = {}
        df = None
        y = None

        if os.path.exists(cache_path):
            cached = np.load(cache_path, allow_pickle=True)
            if "manifest" not in cached:
                print(f"Stale cache (no manifest) at {cache_path}, ignoring.")
            else:
                manifest = json.loads(str(cached["manifest"]))
                df = pd.DataFrame(list(cached["df_records"]))
                mask = self._filter_mask(df)
                df = df.loc[mask].reset_index(drop=True)
                y = cached["y"][mask]
                for name in manifest:
                    cached_blocks[name] = cached[f"block_{name}"][mask]
                cached_blocks.pop("struct", None)
                manifest.pop("struct", None)
                print(f"Cache hit: {list(cached_blocks.keys())} blocks, {y.shape[0]} rows")

        missing = [name for name in requested if name not in manifest]

        if missing:
            if df is None:
                print(f"Building base data from {self.csv_path}")
                df, stockfish_features = load_data(
                    self.csv_path,
                    stockfish_path=self.stockfish_path,
                    num_rows=self.max_rows,
                )
                mask = self._filter_mask(df)
                if stockfish_features is not None:
                    stockfish_features = stockfish_features[mask]
                df = df.loc[mask].reset_index(drop=True)
                y = df["Rating"].values.astype(np.float32)
            else:
                stockfish_features = None

            print(f"Computing missing blocks: {missing}")
            for name in missing:
                block = self._compute_block(name, df, y, stockfish_features, train_idx, val_idx)
                if block is not None:
                    cached_blocks[name] = block[:len(df)].astype(np.float32)
                    manifest[name] = cached_blocks[name].shape[1]
                    print(f"  {name}: {manifest[name]} features")

            save_dict = {
                "y": y,
                "df_records": np.array(df.to_dict("records"), dtype=object).reshape(-1),
                "manifest": np.array(json.dumps(manifest)),
            }
            for name, block in cached_blocks.items():
                save_dict[f"block_{name}"] = block
            np.savez_compressed(cache_path, **save_dict)
            print(f"Saved updated cache to {cache_path}")

        X = np.concatenate([cached_blocks[name] for name in requested if name in cached_blocks], axis=1)
        feature_dims = {name: cached_blocks[name].shape[1] for name in requested if name in cached_blocks}
        for name, dim in feature_dims.items():
            print(f"  {name}: {dim} features")
        print(f"Total features: {X.shape[1]}")
        if mlflow.active_run():
            mlflow.log_params({f"features_{name}": dim for name, dim in feature_dims.items()})
            mlflow.log_param("num_features", X.shape[1])
        return X, y, df

    def load_maia2_only(self):
        return load_maia2_features(self.data_file_name, self.data_dir)

