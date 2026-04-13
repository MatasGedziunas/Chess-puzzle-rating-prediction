import os
import sys
import argparse
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))

from dataset.chess_puzzle_dataset import ChessPuzzleDataset
from dataset.loaders import _compute_correct_move_rank, _reduce_move_elo, _derive_maia2_extended_features
from dataset.maia1_probs import _derive_flat_features

MAIA1_ELOS = [1100, 1300, 1500, 1700, 1900]
MAIA2_ELOS = [1100, 1300, 1500, 1700, 1900]
SPECIALISTS_DIR = "./models/specialists"


def parse_maia_source(maia_source):
    parts = maia_source.lower().split("-")
    if len(parts) < 3 or parts[0] != "maia":
        raise ValueError(
            f"Invalid maia_source: '{maia_source}'. "
            "Expected format: maia-1-1100 | maia-2-1100 | maia-2-rapid-1100 | maia-2-blitz-1100"
        )
    version = int(parts[1])
    if version == 1:
        elo = int(parts[2])
        if elo not in MAIA1_ELOS:
            raise ValueError(f"ELO {elo} not in {MAIA1_ELOS}")
        return 1, None, elo
    elif version == 2:
        if parts[2] in ("rapid", "blitz"):
            model_type = parts[2]
            elo = int(parts[3]) if len(parts) > 3 else None
            if elo is not None and elo not in MAIA2_ELOS:
                raise ValueError(f"ELO {elo} not in {MAIA2_ELOS}")
            return 2, [model_type], elo
        else:
            elo = int(parts[2])
            if elo not in MAIA2_ELOS:
                raise ValueError(f"ELO {elo} not in {MAIA2_ELOS}")
            return 2, ["rapid", "blitz"], elo
    raise ValueError(f"Unsupported maia version: {version}")


def load_maia1_elo_features(data_file_name, elo, data_dir="./data"):
    elo_idx = MAIA1_ELOS.index(elo)
    probs_path = os.path.join(data_dir, f"{data_file_name}_maia1_probs.npy")
    if not os.path.exists(probs_path):
        raise FileNotFoundError(f"Maia1 probs not found: {probs_path}")

    probs = np.load(probs_path, mmap_mode="r")[:, :, elo_idx:elo_idx + 1]
    feature_parts = [_derive_flat_features(probs)]

    top5p_path = os.path.join(data_dir, f"{data_file_name}_maia1_top5_probs.npy")
    top5i_path = os.path.join(data_dir, f"{data_file_name}_maia1_top5_indices.npy")
    pidx_path = os.path.join(data_dir, f"{data_file_name}_maia1_policy_indices.npy")

    if all(os.path.exists(p) for p in [top5p_path, top5i_path, pidx_path]):
        top5_probs = np.load(top5p_path, mmap_mode="r")[:, :, elo_idx:elo_idx + 1, :]
        top5_indices = np.load(top5i_path, mmap_mode="r")[:, :, elo_idx:elo_idx + 1, :]
        policy_indices = np.load(pidx_path, mmap_mode="r")

        rank = _compute_correct_move_rank(probs, top5_indices, policy_indices)
        gap_to_top1 = top5_probs[:, :, :, 0] - probs
        prob_ratio = probs / (top5_probs[:, :, :, 0] + 1e-7)
        feature_parts += [
            _reduce_move_elo(rank),
            _reduce_move_elo(gap_to_top1),
            _reduce_move_elo(prob_ratio),
        ]

    return np.concatenate(feature_parts, axis=1).astype(np.float32)


def load_maia2_elo_features(data_file_name, elo, model_types, data_dir="./data"):
    elo_idx = MAIA2_ELOS.index(elo)
    parts = []

    for model_type in model_types:
        probs_path = os.path.join(data_dir, f"{data_file_name}_maia2_{model_type}_probs.npy")
        if not os.path.exists(probs_path):
            print(f"Warning: {probs_path} not found, skipping {model_type}")
            continue

        probs = np.load(probs_path, mmap_mode="r")[:, :, elo_idx:elo_idx + 1]
        top5p_path = os.path.join(data_dir, f"{data_file_name}_maia2_{model_type}_top5_probs.npy")
        top5i_path = os.path.join(data_dir, f"{data_file_name}_maia2_{model_type}_top5_indices.npy")
        pidx_path = os.path.join(data_dir, f"{data_file_name}_maia2_{model_type}_policy_indices.npy")

        top5_probs = np.load(top5p_path, mmap_mode="r")[:, :, elo_idx:elo_idx + 1, :]
        top5_indices = np.load(top5i_path, mmap_mode="r")[:, :, elo_idx:elo_idx + 1, :]
        policy_indices = np.load(pidx_path, mmap_mode="r")

        features = _derive_maia2_extended_features(
            probs, top5_probs, top5_indices, policy_indices
        )
        parts.append(features)

    if not parts:
        raise FileNotFoundError(f"No maia2 prob files found for model_types={model_types} in {data_dir}")

    return np.concatenate(parts, axis=1).astype(np.float32)


def train_lightgbm(X_train, y_train, X_val, y_val, sample_weights=None, device="cuda"):
    evals_result = {}
    params = {
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "objective": "regression",
        "metric": "mse",
        "random_state": 42,
        "verbosity": -1,
        "device": device,
        "max_bin": 255,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        callbacks=[
            lgb.record_evaluation(evals_result),
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    return model, params, evals_result


def evaluate_model(model, X_eval, y_eval):
    preds = model.predict(X_eval)
    mse = mean_squared_error(y_eval, preds)
    rmse = np.sqrt(mse)
    return mse, rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maia_sources",
        nargs="+",
        default=[],
        help="One or more maia sources whose features are concatenated. "
             "Examples: maia-1-1100  maia-2-1100  maia-2-rapid-1100",
    )
    parser.add_argument(
        "--use_specialist_maia_features",
        action="store_true",
        default=False,
        help="Load Maia specialist features from --maia_sources instead of using the standard dataset maia1/maia2 blocks.",
    )
    parser.add_argument("--csv_path", default="../filtered.csv")
    parser.add_argument("--stockfish_path", default="../filtered_sf_evals.csv")
    parser.add_argument("--themes_csv_path", default="../filtered_themes_only.csv")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--filter_rating_deviation", action="store_true", default=True)
    parser.add_argument("--use_sample_weights", action="store_true", default=False)
    parser.add_argument("--device", default="cuda", help="Device for LightGBM: cuda or cpu")
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=["struct", "themes", "advanced", "stockfish", "maia1", "maia2"],
        help="Explicit list of base feature blocks to include. "
             "Available: struct themes advanced stockfish maia1 maia2 maia2_mlp. "
             "Overrides the default flag-based selection when set.",
    )
    parser.add_argument(
        "--splits_path",
        default=None,
        help="Path to .npz file with train_idx/val_idx/test_idx (e.g. ./data/filtered_splits.npz). "
             "If not provided, splits are recomputed with the default random seed.",
    )
    args = parser.parse_args()

    if args.use_specialist_maia_features and not args.maia_sources:
        parser.error("--use_specialist_maia_features requires --maia_sources")

    specialist_label = "__".join(s.replace("-", "_") for s in args.maia_sources) if args.maia_sources else "standard"
    blocks_label = "_".join(dataset_blocks) if 'dataset_blocks' in locals() else ""
    model_label = specialist_label if args.use_specialist_maia_features else f"base__{blocks_label}"
    data_file_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    dataset_blocks = [block for block in args.blocks if block not in {"maia1", "maia2"}] if args.use_specialist_maia_features else args.blocks
    blocks_label = "_".join(dataset_blocks)
    model_label = specialist_label if args.use_specialist_maia_features else f"base__{blocks_label}"

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Chess_Puzzle_Maia_Specialists")

    dataset = ChessPuzzleDataset(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        stockfish_path=args.stockfish_path,
        themes_csv_path=args.themes_csv_path,
        use_maia1=not args.use_specialist_maia_features and "maia1" in args.blocks,
        use_maia2=not args.use_specialist_maia_features and "maia2" in args.blocks,
        use_maia2_mlp=False,
        filter_rating_deviation=args.filter_rating_deviation,
        max_rows=args.max_rows,
        blocks=dataset_blocks,
    )

    X, y, df = dataset.load()

    if args.use_specialist_maia_features:
        specialist_features = []
        for source in args.maia_sources:
            version, model_types, elo = parse_maia_source(source)
            if version == 1:
                specialist_features.append(load_maia1_elo_features(data_file_name, elo, args.data_dir))
            else:
                specialist_features.append(load_maia2_elo_features(data_file_name, elo, model_types, args.data_dir))
        X = np.concatenate([X] + specialist_features, axis=1).astype(np.float32)

    n = len(df)
    rating_deviation = df["RatingDeviation"].values.astype(np.float32) if "RatingDeviation" in df.columns else None
    sample_weights = None
    if args.use_sample_weights and rating_deviation is not None:
        sample_weights = (1.0 / np.maximum(rating_deviation, 1.0)).astype(np.float32)

    if args.splits_path and os.path.exists(args.splits_path):
        splits = np.load(args.splits_path)
        train_idx = splits["train_idx"]
        val_idx = splits["val_idx"]
        test_idx = splits["test_idx"]
        print(f"Loaded splits from {args.splits_path}")
    else:
        indices = np.arange(n)
        train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=1.0 / 9.0, random_state=42)

    row_count = len(df)
    X = X[:row_count]
    total_features = X.shape[1]
    print(f"Preparing split matrices with {total_features} features across {row_count} rows ")

    X_train = X[train_idx].astype(np.float32)
    X_val = X[val_idx].astype(np.float32)
    X_test = X[test_idx].astype(np.float32)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    train_weights = sample_weights[train_idx] if sample_weights is not None else None

    del X
    gc.collect()

    print(
        f"specialist_maia={args.use_specialist_maia_features}  maia_sources={args.maia_sources}  features={total_features}  "
        f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}"
    )

    interrupted = False
    with mlflow.start_run() as run:
        mlflow.log_params(vars(args))
        mlflow.log_param("model_label", model_label)
        mlflow.log_param("blocks_str", " ".join(dataset_blocks))
        mlflow.log_param("num_train", len(X_train))
        mlflow.log_param("num_val", len(X_val))
        mlflow.log_param("num_test", len(X_test))
        mlflow.log_param("use_specialist_maia_features", args.use_specialist_maia_features)
        if args.use_specialist_maia_features:
            mlflow.log_param("specialist_maia_sources", " ".join(args.maia_sources))

        model = None
        model_params = {}
        evals_result = {}
        try:
            model, model_params, evals_result = train_lightgbm(X_train, y_train, X_val, y_val, train_weights, args.device)
        except KeyboardInterrupt:
            interrupted = True
            print("\nTraining interrupted. Saving partial results...")

        for k, v in model_params.items():
            mlflow.log_param(k, v)
        if interrupted:
            mlflow.log_param("interrupted", True)

        train_curve = evals_result.get("train", {}).get("l2", [])
        val_curve = evals_result.get("val", {}).get("l2", [])
        for step, mse in enumerate(train_curve):
            mlflow.log_metric("train_mse_iter", mse, step=step)
        for step, mse in enumerate(val_curve):
            mlflow.log_metric("val_mse_iter", mse, step=step)

        if model is None:
            raise RuntimeError("Training did not produce a model.")

        train_mse, train_rmse = evaluate_model(model, X_train, y_train)
        val_mse, val_rmse = evaluate_model(model, X_val, y_val)
        test_mse, test_rmse = evaluate_model(model, X_test, y_test)

        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)

        mlflow.lightgbm.log_model(model, f"lgbm_{model_label}")

        os.makedirs(SPECIALISTS_DIR, exist_ok=True)
        model_path = os.path.join(SPECIALISTS_DIR, f"{model_label}_lgbm.pkl")
        joblib.dump(model, model_path)

        status = "interrupted" if interrupted else "complete"
        print(f"\nTraining {status}.")
        print(f"  Train MSE: {train_mse:.2f} (RMSE: {train_rmse:.2f})")
        print(f"  Val   MSE: {val_mse:.2f} (RMSE: {val_rmse:.2f})")
        print(f"  Test  MSE: {test_mse:.2f} (RMSE: {test_rmse:.2f})")
        print(f"  Saved model -> {model_path}")
        print(f"  MLflow run  -> {run.info.run_id}")

        results_dir = f"./results/{data_file_name}"
        os.makedirs(results_dir, exist_ok=True)
        blocks_suffix = "_".join(dataset_blocks)
        pd.DataFrame([{
            "maia_sources": " ".join(args.maia_sources),
            "model_label": model_label,
            "blocks": " ".join(dataset_blocks),
            "use_specialist_maia_features": args.use_specialist_maia_features,
            "interrupted": interrupted,
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "val_mse": val_mse,
            "val_rmse": val_rmse,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "model_path": model_path,
            "mlflow_run_id": run.info.run_id,
        }]).to_csv(f"{results_dir}/specialist_{model_label}__{blocks_suffix}.csv", index=False)
