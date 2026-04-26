import os
import sys
import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

CURRENT_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(CURRENT_DIR, "dataset")
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if DATASET_DIR not in sys.path:
    sys.path.insert(0, DATASET_DIR)

from dataset.chess_puzzle_dataset import ChessPuzzleDataset

DEFAULT_CSV_PATH = os.path.normpath(os.path.join(CURRENT_DIR, "..", "filtered.csv"))
DEFAULT_STOCKFISH_PATH = os.path.normpath(os.path.join(CURRENT_DIR, "..", "filtered_sf_evals.csv"))
DEFAULT_THEMES_PATH = os.path.normpath(os.path.join(CURRENT_DIR, "..", "filtered_themes_only.csv"))
DEFAULT_DATA_DIR = os.path.join(CURRENT_DIR, "data")
MODEL_DIR = os.path.join(CURRENT_DIR, "models", "model_comparison")
MLFLOW_TRACKING_DIR = Path(os.path.join(CURRENT_DIR, "mlruns_model_comparison")).resolve().as_uri()
EXPERIMENT_NAME = "Chess_Puzzle_CatBoost_Full_Dataset"
DEFAULT_BLOCKS = ["struct", "themes", "advanced", "stockfish", "maia1", "maia2"]
RANDOM_STATE = 42
BOOSTING_ROUNDS = 500000
LOG_PERIOD = 100


def configure_cuda_device(cuda_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)


def build_catboost():
    params = {
        "iterations": BOOSTING_ROUNDS,
        "early_stopping_rounds": 200,
        "learning_rate": 0.05,
        "depth": 6,
        "min_data_in_leaf": 20,
        "border_count": 255,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": RANDOM_STATE,
        "verbose": LOG_PERIOD,
        "task_type": "GPU",
        "devices": "0",
    }
    model = CatBoostRegressor(**params)
    return model, params


def evaluate_model(model, X_eval, y_eval):
    predictions = model.predict(X_eval)
    mse = mean_squared_error(y_eval, predictions)
    rmse = float(np.sqrt(mse))
    return mse, rmse


def save_model(model, model_name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{model_name}.cbm")
    model.save_model(model_path)
    return model_path


def build_results_dir(data_file_name):
    return os.path.join(CURRENT_DIR, "results", data_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--stockfish_path", default=DEFAULT_STOCKFISH_PATH)
    parser.add_argument("--themes_csv_path", default=DEFAULT_THEMES_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--filter_rating_deviation", action="store_true", default=True)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=DEFAULT_BLOCKS,
        help="Feature blocks to include. Available: struct themes advanced stockfish maia1 maia2 maia2_mlp",
    )
    parser.add_argument(
        "--splits_path",
        default=None,
        help="Path to .npz file with train_idx/val_idx/test_idx. If not provided, splits are recomputed with the default random seed.",
    )
    args = parser.parse_args()

    configure_cuda_device(args.cuda_device)
    data_file_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    blocks_label = "_".join(args.blocks)
    model_name = f"catboost_full_dataset__{data_file_name}__{blocks_label}"
    max_rows_label = args.max_rows if args.max_rows is not None else "all"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR)
    mlflow.set_experiment(EXPERIMENT_NAME)

    dataset = ChessPuzzleDataset(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        stockfish_path=args.stockfish_path,
        themes_csv_path=args.themes_csv_path,
        use_maia1="maia1" in args.blocks,
        use_maia2="maia2" in args.blocks,
        use_maia2_mlp="maia2_mlp" in args.blocks,
        filter_rating_deviation=args.filter_rating_deviation,
        max_rows=args.max_rows,
        blocks=args.blocks,
    )

    X, y, df = dataset.load()
    row_count = len(df)
    X = X[:row_count]
    y = y[:row_count]

    if args.splits_path and os.path.exists(args.splits_path):
        splits = np.load(args.splits_path)
        train_idx = splits["train_idx"]
        val_idx = splits["val_idx"]
        test_idx = splits["test_idx"]
        print(f"Loaded splits from {args.splits_path}")
    else:
        indices = np.arange(row_count)
        train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=RANDOM_STATE)
        train_idx, val_idx = train_test_split(train_idx, test_size=1.0 / 9.0, random_state=RANDOM_STATE)

    X_train = X[train_idx].astype(np.float32)
    X_val = X[val_idx].astype(np.float32)
    X_test = X[test_idx].astype(np.float32)
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    total_features = X.shape[1]

    print(
        f"features={total_features} rows={row_count} blocks={' '.join(args.blocks)} "
        f"cuda_device={args.cuda_device} "
        f"train={len(X_train)} val={len(X_val)} test={len(X_test)}"
    )

    results_dir = build_results_dir(data_file_name)
    os.makedirs(results_dir, exist_ok=True)

    with mlflow.start_run(run_name=model_name, nested=False) as run:
        mlflow.log_params(vars(args))
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("blocks_str", " ".join(args.blocks))
        mlflow.log_param("num_features", total_features)
        mlflow.log_param("num_rows", row_count)
        mlflow.log_param("num_train", len(X_train))
        mlflow.log_param("num_val", len(X_val))
        mlflow.log_param("num_test", len(X_test))
        mlflow.log_param("training_scheme", "train_val_test_no_cv")

        model, model_params = build_catboost()
        mlflow.log_params({f"model_{key}": value for key, value in model_params.items()})

        train_start_time = time.perf_counter()
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
        )
        training_time_seconds = time.perf_counter() - train_start_time

        train_mse, train_rmse = evaluate_model(model, X_train, y_train)
        val_mse, val_rmse = evaluate_model(model, X_val, y_val)
        test_mse, test_rmse = evaluate_model(model, X_test, y_test)

        mlflow.log_metric("training_time_seconds", training_time_seconds)
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.catboost.log_model(model, model_name)

        model_path = save_model(model, model_name)

        results_path = os.path.join(
            results_dir,
            f"catboost_full_dataset_results_{max_rows_label}_{blocks_label}.csv",
        )
        pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "training_scheme": "train_val_test_no_cv",
                    "blocks": " ".join(args.blocks),
                    "num_rows": row_count,
                    "num_train": len(X_train),
                    "num_val": len(X_val),
                    "num_test": len(X_test),
                    "num_features": total_features,
                    "training_time_seconds": training_time_seconds,
                    "train_mse": train_mse,
                    "train_rmse": train_rmse,
                    "val_mse": val_mse,
                    "val_rmse": val_rmse,
                    "test_mse": test_mse,
                    "test_rmse": test_rmse,
                    "model_path": model_path,
                    "mlflow_run_id": run.info.run_id,
                }
            ]
        ).to_csv(results_path, index=False)

        print("\nTraining complete.")
        print(f"  Train MSE: {train_mse:.2f}")
        print(f"  Train RMSE: {train_rmse:.2f}")
        print(f"  Val   MSE: {val_mse:.2f}")
        print(f"  Val   RMSE: {val_rmse:.2f}")
        print(f"  Test  MSE: {test_mse:.2f}")
        print(f"  Test  RMSE: {test_rmse:.2f}")
        print(f"  Training time: {training_time_seconds:.2f}s")
        print(f"  Saved model -> {model_path}")
        print(f"  Saved results -> {results_path}")
        print(f"  MLflow run  -> {run.info.run_id}")
