import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

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
EXPERIMENT_NAME = "Chess_Puzzle_Model_Comparison"
DEFAULT_MODELS = ["lightgbm", "random_forest", "catboost", "mlp"]
DEFAULT_BLOCKS = ["struct", "themes", "advanced", "stockfish", "maia1", "maia2"]
RANDOM_STATE = 42
BOOSTING_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 100
RANDOM_FOREST_TREES = 500
MLP_MAX_ITER = 5000
LOG_PERIOD = 100
CV_FOLDS = 5


def evaluate_model(model, X_eval, y_eval):
    predictions = model.predict(X_eval)
    mse = mean_squared_error(y_eval, predictions)
    rmse = float(np.sqrt(mse))
    return mse, rmse


def configure_cuda_device(cuda_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)


def build_cv_splits(row_count):
    indices = np.arange(row_count)
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(kfold.split(indices))


def build_lightgbm():
    params = {
        "n_estimators": BOOSTING_ROUNDS,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "objective": "regression",
        "metric": ["mse", "rmse"],
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "device": "cuda",
        "max_bin": 255,
    }
    model = lgb.LGBMRegressor(**params)
    return model, params


def build_random_forest():
    params = {
        "n_estimators": RANDOM_FOREST_TREES,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    model = RandomForestRegressor(**params)
    return model, params


def build_catboost():
    if CatBoostRegressor is None:
        raise ImportError("catboost is not installed. Install it or remove catboost from --models.")
    params = {
        "iterations": BOOSTING_ROUNDS,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_seed": RANDOM_STATE,
        "verbose": LOG_PERIOD,
        "task_type": "GPU",
    }
    model = CatBoostRegressor(**params)
    return model, params


def build_mlp():
    params = {
        "hidden_layer_sizes": (512, 256, 128),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": 256,
        "learning_rate_init": 1e-3,
        "max_iter": MLP_MAX_ITER,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": EARLY_STOPPING_ROUNDS,
        "random_state": RANDOM_STATE,
        "verbose": True,
    }
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(**params)),
        ]
    )
    return model, params


def get_model_builder(model_name):
    builders = {
        "lightgbm": lambda: build_lightgbm(),
        "random_forest": build_random_forest,
        "catboost": build_catboost,
        "mlp": build_mlp,
    }
    return builders[model_name]


def fit_model(model_name, model, X_train, y_train, X_val, y_val):
    if model_name == "lightgbm":
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=LOG_PERIOD),
        ]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=["train", "val"],
            callbacks=callbacks,
        )
        return

    if model_name == "catboost":
        fit_kwargs = {
            "X": X_train,
            "y": y_train,
            "eval_set": (X_val, y_val),
            "use_best_model": True,
        }
        model.fit(**fit_kwargs)
        return

    if model_name == "random_forest":
        fit_kwargs = {
            "X": X_train,
            "y": y_train,
        }
        model.fit(**fit_kwargs)
        return

    model.fit(X_train, y_train)


def log_model_to_mlflow(model_name, model):
    if model_name == "lightgbm":
        mlflow.lightgbm.log_model(model, model_name)
        return

    if model_name == "catboost":
        mlflow.catboost.log_model(model, model_name)
        return

    mlflow.sklearn.log_model(model, model_name)


def save_model(model_name, model, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    if model_name == "catboost":
        model_path = os.path.join(model_dir, f"{model_name}.cbm")
        model.save_model(model_path)
        return model_path

    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
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
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=DEFAULT_MODELS,
    )
    args = parser.parse_args()

    if "catboost" in args.models and CatBoostRegressor is None:
        raise ImportError("catboost is not installed. Install it or run without catboost in --models.")

    configure_cuda_device(args.cuda_device)
    data_file_name = os.path.splitext(os.path.basename(args.csv_path))[0]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR)
    mlflow.set_experiment(EXPERIMENT_NAME)

    dataset = ChessPuzzleDataset(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        stockfish_path=args.stockfish_path,
        themes_csv_path=args.themes_csv_path,
        use_maia1=True,
        use_maia2=True,
        use_maia2_mlp=False,
        filter_rating_deviation=args.filter_rating_deviation,
        max_rows=args.max_rows,
        blocks=DEFAULT_BLOCKS,
    )

    X, y, df = dataset.load()
    row_count = len(df)
    X = X[:row_count].astype(np.float32)
    y = y[:row_count]
    cv_splits = build_cv_splits(row_count)

    print(
        f"features={X.shape[1]} rows={row_count} folds={CV_FOLDS} "
        f"models={' '.join(args.models)} cuda_device={args.cuda_device}"
    )

    results = []
    results_dir = build_results_dir(data_file_name)
    os.makedirs(results_dir, exist_ok=True)

    for model_name in args.models:
        print(f"\nTraining {model_name}...")

        with mlflow.start_run(run_name=model_name, nested=False) as run:
            mlflow.log_params(vars(args))
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("blocks_str", " ".join(DEFAULT_BLOCKS))
            mlflow.log_param("num_features", X.shape[1])
            mlflow.log_param("num_rows", row_count)
            mlflow.log_param("cv_folds", CV_FOLDS)

            fold_results = []
            model_params = None
            last_model = None

            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
                X_train = X[train_idx]
                X_val = X[val_idx]
                y_train = y[train_idx]
                y_val = y[val_idx]

                builder = get_model_builder(model_name)
                model, current_model_params = builder()
                if model_params is None:
                    model_params = current_model_params
                    mlflow.log_params({f"model_{key}": value for key, value in model_params.items()})

                print(
                    f"  Fold {fold_idx}/{CV_FOLDS}: "
                    f"train={len(X_train)} val={len(X_val)}"
                )

                train_start_time = time.perf_counter()
                fit_model(model_name, model, X_train, y_train, X_val, y_val)
                training_time_seconds = time.perf_counter() - train_start_time

                train_mse, train_rmse = evaluate_model(model, X_train, y_train)
                val_mse, val_rmse = evaluate_model(model, X_val, y_val)

                mlflow.log_metric(f"fold_{fold_idx}_training_time_seconds", training_time_seconds)
                mlflow.log_metric(f"fold_{fold_idx}_train_mse", train_mse)
                mlflow.log_metric(f"fold_{fold_idx}_train_rmse", train_rmse)
                mlflow.log_metric(f"fold_{fold_idx}_val_mse", val_mse)
                mlflow.log_metric(f"fold_{fold_idx}_val_rmse", val_rmse)

                model_path = save_model(f"{model_name}_fold{fold_idx}", model, MODEL_DIR)
                last_model = model

                print(f"    Training time: {training_time_seconds:.2f}s")
                print(f"    Train MSE: {train_mse:.2f} (RMSE: {train_rmse:.2f})")
                print(f"    Val   MSE: {val_mse:.2f} (RMSE: {val_rmse:.2f})")
                print(f"    Saved model -> {model_path}")

                fold_results.append(
                    {
                        "model_name": model_name,
                        "fold": fold_idx,
                        "training_time_seconds": training_time_seconds,
                        "train_mse": train_mse,
                        "train_rmse": train_rmse,
                        "val_mse": val_mse,
                        "val_rmse": val_rmse,
                        "model_path": model_path,
                        "mlflow_run_id": run.info.run_id,
                    }
                )

            if last_model is not None:
                log_model_to_mlflow(model_name, last_model)

            fold_df = pd.DataFrame(fold_results)
            best_fold_idx = fold_df["val_rmse"].idxmin()
            worst_fold_idx = fold_df["val_rmse"].idxmax()
            best_fold = fold_df.loc[best_fold_idx]
            worst_fold = fold_df.loc[worst_fold_idx]
            summary = {
                "model_name": model_name,
                "training_time_seconds_mean": fold_df["training_time_seconds"].mean(),
                "training_time_seconds_std": fold_df["training_time_seconds"].std(ddof=0),
                "train_mse_mean": fold_df["train_mse"].mean(),
                "train_mse_std": fold_df["train_mse"].std(ddof=0),
                "train_rmse_mean": fold_df["train_rmse"].mean(),
                "train_rmse_std": fold_df["train_rmse"].std(ddof=0),
                "val_mse_mean": fold_df["val_mse"].mean(),
                "val_mse_std": fold_df["val_mse"].std(ddof=0),
                "val_rmse_mean": fold_df["val_rmse"].mean(),
                "val_rmse_std": fold_df["val_rmse"].std(ddof=0),
                "best_fold": int(best_fold["fold"]),
                "best_fold_training_time_seconds": best_fold["training_time_seconds"],
                "best_fold_train_mse": best_fold["train_mse"],
                "best_fold_train_rmse": best_fold["train_rmse"],
                "best_fold_val_mse": best_fold["val_mse"],
                "best_fold_val_rmse": best_fold["val_rmse"],
                "worst_fold": int(worst_fold["fold"]),
                "worst_fold_training_time_seconds": worst_fold["training_time_seconds"],
                "worst_fold_train_mse": worst_fold["train_mse"],
                "worst_fold_train_rmse": worst_fold["train_rmse"],
                "worst_fold_val_mse": worst_fold["val_mse"],
                "worst_fold_val_rmse": worst_fold["val_rmse"],
                "mlflow_run_id": run.info.run_id,
            }

            mlflow.log_metric("training_time_seconds_mean", summary["training_time_seconds_mean"])
            mlflow.log_metric("training_time_seconds_std", summary["training_time_seconds_std"])
            mlflow.log_metric("train_mse_mean", summary["train_mse_mean"])
            mlflow.log_metric("train_mse_std", summary["train_mse_std"])
            mlflow.log_metric("train_rmse_mean", summary["train_rmse_mean"])
            mlflow.log_metric("train_rmse_std", summary["train_rmse_std"])
            mlflow.log_metric("val_mse_mean", summary["val_mse_mean"])
            mlflow.log_metric("val_mse_std", summary["val_mse_std"])
            mlflow.log_metric("val_rmse_mean", summary["val_rmse_mean"])
            mlflow.log_metric("val_rmse_std", summary["val_rmse_std"])
            mlflow.log_metric("best_fold", summary["best_fold"])
            mlflow.log_metric("best_fold_training_time_seconds", summary["best_fold_training_time_seconds"])
            mlflow.log_metric("best_fold_train_mse", summary["best_fold_train_mse"])
            mlflow.log_metric("best_fold_train_rmse", summary["best_fold_train_rmse"])
            mlflow.log_metric("best_fold_val_mse", summary["best_fold_val_mse"])
            mlflow.log_metric("best_fold_val_rmse", summary["best_fold_val_rmse"])
            mlflow.log_metric("worst_fold", summary["worst_fold"])
            mlflow.log_metric("worst_fold_training_time_seconds", summary["worst_fold_training_time_seconds"])
            mlflow.log_metric("worst_fold_train_mse", summary["worst_fold_train_mse"])
            mlflow.log_metric("worst_fold_train_rmse", summary["worst_fold_train_rmse"])
            mlflow.log_metric("worst_fold_val_mse", summary["worst_fold_val_mse"])
            mlflow.log_metric("worst_fold_val_rmse", summary["worst_fold_val_rmse"])

            print(f"  Mean training time: {summary['training_time_seconds_mean']:.2f}s +/- {summary['training_time_seconds_std']:.2f}s")
            print(f"  Mean train RMSE: {summary['train_rmse_mean']:.2f} +/- {summary['train_rmse_std']:.2f}")
            print(f"  Mean val   RMSE: {summary['val_rmse_mean']:.2f} +/- {summary['val_rmse_std']:.2f}")
            print(
                f"  Best fold: {summary['best_fold']} "
                f"(val RMSE: {summary['best_fold_val_rmse']:.2f}, train RMSE: {summary['best_fold_train_rmse']:.2f})"
            )
            print(
                f"  Worst fold: {summary['worst_fold']} "
                f"(val RMSE: {summary['worst_fold_val_rmse']:.2f}, train RMSE: {summary['worst_fold_train_rmse']:.2f})"
            )
            print(f"  MLflow run  -> {run.info.run_id}")

            results.extend(fold_results)
            results.append(summary)

    results_df = pd.DataFrame(results)
    fold_results_df = results_df[results_df.get("fold").notna()].copy()
    summary_df = results_df[results_df.get("fold").isna()].copy()
    results_path = os.path.join(results_dir, f"model_comparison_results_{args.max_rows}.csv")
    fold_results_path = os.path.join(results_dir, f"model_comparison_fold_results_{args.max_rows}.csv")
    summary_df = summary_df.sort_values("val_rmse_mean").reset_index(drop=True)
    fold_results_df.to_csv(fold_results_path, index=False)
    summary_df.to_csv(results_path, index=False)

    print("\nComparison complete.")
    print(summary_df[["model_name", "training_time_seconds_mean", "val_rmse_mean", "val_rmse_std"]].to_string(index=False))
    print(f"Saved summary -> {results_path}")
    print(f"Saved fold results -> {fold_results_path}")
