import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import mlflow
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))

from dataset.chess_puzzle_dataset import ChessPuzzleDataset
from lightgbm_maia_specialist import parse_maia_source, load_maia1_elo_features, load_maia2_elo_features

SPECIALISTS_DIR = "./models/specialists"


def label_to_sources(model_filename):
    label = model_filename.replace("_lgbm.pkl", "")
    return [s.replace("_", "-") for s in label.split("__")]


def load_specialists(specialists_dir):
    specialists = {}
    for fname in sorted(os.listdir(specialists_dir)):
        if not fname.endswith("_lgbm.pkl"):
            continue
        path = os.path.join(specialists_dir, fname)
        label = fname.replace("_lgbm.pkl", "")
        sources = label_to_sources(fname)
        specialists[label] = (joblib.load(path), sources)
        print(f"  Loaded: {fname}  sources={sources}")
    return specialists


def build_specialist_X(sources, data_file_name, X_base, data_dir):
    row_count = len(X_base)
    parts = []
    for source in sources:
        version, model_types, elo = parse_maia_source(source)
        if version == 1:
            parts.append(load_maia1_elo_features(data_file_name, elo, data_dir)[:row_count])
        else:
            parts.append(load_maia2_elo_features(data_file_name, elo, model_types, data_dir)[:row_count])
    return np.concatenate([X_base] + parts, axis=1)


def collect_predictions(specialists, data_file_name, X_base, data_dir):
    preds = {}
    for label, (model, sources) in specialists.items():
        X = build_specialist_X(sources, data_file_name, X_base, data_dir)
        preds[label] = model.predict(X).astype(np.float32)
        print(f"  {label}: {preds[label].shape[0]} predictions")
    return preds


def build_meta_features(predictions, X_base):
    pred_matrix = np.column_stack(list(predictions.values()))
    pred_mean = pred_matrix.mean(axis=1, keepdims=True)
    pred_std = pred_matrix.std(axis=1, keepdims=True)
    pred_min = pred_matrix.min(axis=1, keepdims=True)
    pred_max = pred_matrix.max(axis=1, keepdims=True)
    return np.concatenate(
        [X_base, pred_matrix, pred_mean, pred_std, pred_min, pred_max], axis=1
    ).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="../filtered.csv")
    parser.add_argument("--stockfish_path", default="../filtered_sf_evals.csv")
    parser.add_argument("--themes_csv_path", default="../filtered_themes_only.csv")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--specialists_dir", default=SPECIALISTS_DIR)
    parser.add_argument("--splits_path", default="./data/filtered_splits.npz")
    parser.add_argument("--filter_rating_deviation", action="store_true", default=True)
    parser.add_argument("--task_type", default="GPU", choices=["GPU", "CPU"])
    args = parser.parse_args()

    data_file_name = os.path.splitext(os.path.basename(args.csv_path))[0]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Chess_Puzzle_Maia_Specialists")

    dataset = ChessPuzzleDataset(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        stockfish_path=args.stockfish_path,
        themes_csv_path=args.themes_csv_path,
        use_maia1=False,
        use_maia2=False,
        use_maia2_mlp=False,
        filter_rating_deviation=args.filter_rating_deviation,
        max_rows=None,
    )

    X_base, y, df = dataset.load()
    n = len(df)

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

    print(f"\nLoading specialist models from {args.specialists_dir}...")
    specialists = load_specialists(args.specialists_dir)
    if not specialists:
        raise FileNotFoundError(f"No specialist models found in {args.specialists_dir}")

    print(f"\nGenerating predictions from {len(specialists)} specialists...")
    predictions = collect_predictions(specialists, data_file_name, X_base, args.data_dir)

    X_meta = build_meta_features(predictions, X_base)
    print(f"\nMeta features shape: {X_meta.shape}")

    X_train = X_meta[train_idx]
    X_val = X_meta[val_idx]
    X_test = X_meta[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    catboost_params = {
        "iterations": 5000,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "early_stopping_rounds": 100,
        "task_type": args.task_type,
        "verbose": 200,
        "random_seed": 42,
    }

    with mlflow.start_run() as run:
        mlflow.log_param("specialists", " | ".join(specialists.keys()))
        mlflow.log_param("num_specialists", len(specialists))
        mlflow.log_param("meta_features", X_meta.shape[1])
        mlflow.log_param("num_train", len(X_train))
        mlflow.log_param("num_val", len(X_val))
        mlflow.log_param("num_test", len(X_test))
        for k, v in catboost_params.items():
            mlflow.log_param(k, v)

        meta_model = CatBoostRegressor(**catboost_params)
        meta_model.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val), use_best_model=True)

        val_preds = meta_model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_preds)
        val_rmse = np.sqrt(val_mse)

        test_preds = meta_model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_preds)
        test_rmse = np.sqrt(test_mse)

        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)

        os.makedirs("./models", exist_ok=True)
        model_path = "./models/catboost_specialist_ensemble.cbm"
        meta_model.save_model(model_path)

        print(f"\nVal  MSE: {val_mse:.2f} (RMSE: {val_rmse:.2f})")
        print(f"Test MSE: {test_mse:.2f} (RMSE: {test_rmse:.2f})")
        print(f"Saved -> {model_path}")
        print(f"MLflow run -> {run.info.run_id}")

        out_dir = "./results/ensemble"
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([{
            "num_specialists": len(specialists),
            "specialists": " | ".join(specialists.keys()),
            "val_mse": val_mse,
            "val_rmse": val_rmse,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "best_iteration": meta_model.get_best_iteration(),
            "model_path": model_path,
            "mlflow_run_id": run.info.run_id,
        }]).to_csv(f"{out_dir}/catboost_ensemble_results.csv", index=False)
