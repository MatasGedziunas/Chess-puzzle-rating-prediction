import numpy as np
import os
import argparse
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import mlflow

from dataset.loaders import load_data
from dataset.board_features import build_features, encode_themes, build_advanced_features, build_success_prob_features


def flatten_maia_embeddings(maia_seq):
    return maia_seq.mean(axis=1)


def prepare_features(X_struct, X_themes, maia_seq_flat, advanced_features, success_prob_features, stockfish_features=None):
    parts = [X_struct, X_themes, advanced_features, success_prob_features]
    if maia_seq_flat is not None:
        parts.append(maia_seq_flat)
    if stockfish_features is not None:
        parts.append(stockfish_features)
    return np.concatenate(parts, axis=1)


def apply_rating_mask(mask, X_struct, X_themes, maia_seq_flat, stockfish_features, y, df):
    X_struct = X_struct[mask]
    X_themes = X_themes[mask]
    if maia_seq_flat is not None:
        maia_seq_flat = maia_seq_flat[mask]
    if stockfish_features is not None:
        stockfish_features = stockfish_features[mask]
    y = y[mask]
    df = df[mask].reset_index(drop=True)
    return X_struct, X_themes, maia_seq_flat, stockfish_features, y, df


def train_xgboost(X_train, y_train, X_val, y_val, sample_weights=None):
    params = {
        "n_estimators": 5000,
        "learning_rate": 0.1,
        "max_leaves": 63,
        "max_bin": 63,
        "tree_method": "hist",
        "device": "cpu",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "verbosity": 1,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100,
    )
    return model, params


def train_lightgbm(X_train, y_train, X_val, y_val, sample_weights=None):
    params = {
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "objective": "regression",
        "metric": "rmse",
        "random_state": 42,
        "verbosity": -1,
        "device": "cuda",
        "max_bin": 255,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    return model, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_rating", type=int, default=None)
    parser.add_argument("--max_rating", type=int, default=None)
    parser.add_argument("--model_type", choices=["xgboost", "lightgbm"], default="lightgbm")
    parser.add_argument("--max_rows", type=int, default=200000)
    parser.add_argument("--use_maia_embeddings", action="store_true", default=False)
    parser.add_argument("--filter_rating_deviation", action="store_true", default=True)
    parser.add_argument("--use_sample_weights", action="store_true", default=False)
    args = parser.parse_args()

    csv_path = "../filtered.csv"
    embeddings_path = "./data/p200k/maia2.npy" if args.use_maia_embeddings else None
    stockfish_path = "../filtered_sf_evals.csv"

    mlflow.set_experiment("Chess_Puzzle_Rating_Prediction")

    df, maia_seq, stockfish_features = load_data(
        csv_path,
        embeddings_path,
        stockfish_path,
        load_maia_embeddings=args.use_maia_embeddings,
    )

    if args.filter_rating_deviation and 'RatingDeviation' in df.columns:
        df = df[df['RatingDeviation'] <= 90].reset_index(drop=True)

    X_struct = build_features(df)
    X_themes = encode_themes(df, themes_csv_path="../filtered_themes_only.csv")
    maia_seq_flat = flatten_maia_embeddings(maia_seq) if args.use_maia_embeddings else None
    y = df['Rating'].values

    mask = np.ones(len(y), dtype=bool)
    if args.min_rating is not None:
        mask &= (y >= args.min_rating)
    if args.max_rating is not None:
        mask &= (y < args.max_rating)

    X_struct, X_themes, maia_seq_flat, stockfish_features, y, df = apply_rating_mask(
        mask, X_struct, X_themes, maia_seq_flat, stockfish_features, y, df
    )

    df = df.head(args.max_rows)
    n = len(df)
    X_struct = X_struct[:n]
    X_themes = X_themes[:n]
    if maia_seq_flat is not None:
        maia_seq_flat = maia_seq_flat[:n]
    if stockfish_features is not None:
        stockfish_features = stockfish_features[:n]
    y = y[:n]

    rating_deviation = df['RatingDeviation'].values[:n].astype(np.float32) if 'RatingDeviation' in df.columns else None
    sample_weights = None
    if args.use_sample_weights and rating_deviation is not None:
        sample_weights = (1.0 / np.maximum(rating_deviation, 1.0)).astype(np.float32) 

    data_file_name = os.path.splitext(os.path.basename(csv_path))[0]
    advanced_features = build_advanced_features(df, data_file_name)
    success_prob_features = build_success_prob_features(df)

    X = prepare_features(X_struct, X_themes, maia_seq_flat, advanced_features, success_prob_features, stockfish_features)
    print(f"Total feature dimension: {X.shape[1]}")
    print(f"  Struct features:    {X_struct.shape[1]}")
    print(f"  Theme features:     {X_themes.shape[1]}")
    if maia_seq_flat is not None:
        print(f"  Maia embeddings:    {maia_seq_flat.shape[1]}")
    else:
        print(f"  Maia embeddings:    disabled")
    print(f"  Advanced features:  {advanced_features.shape[1]}")
    print(f"  Success prob stats: {success_prob_features.shape[1]}")
    print(f"  Length feature:     included in struct")
    if stockfish_features is not None:
        print(f"  Stockfish features: {stockfish_features.shape[1]}")

    indices = np.arange(n)
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"\nTraining {args.model_type} with {len(X_train)} train / {len(X_val)} val samples")

    model_params = {}
    interrupted = False

    with mlflow.start_run():
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("min_rating", args.min_rating)
        mlflow.log_param("max_rating", args.max_rating)
        mlflow.log_param("filter_rating_deviation", args.filter_rating_deviation)
        mlflow.log_param("use_sample_weights", args.use_sample_weights)
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("num_train", len(X_train))
        mlflow.log_param("num_val", len(X_val))

        try:
            if args.model_type == "xgboost":
                model, model_params = train_xgboost(X_train, y_train, X_val, y_val, sample_weights[train_idx] if sample_weights is not None else None)
            else:
                model, model_params = train_lightgbm(X_train, y_train, X_val, y_val, sample_weights[train_idx] if sample_weights is not None else None)
        except KeyboardInterrupt:
            interrupted = True
            print("\nTraining interrupted. Saving partial results...")

        for k, v in model_params.items():
            mlflow.log_param(k, v)

        if interrupted:
            mlflow.log_param("interrupted", True)

        suffix = ""
        if args.min_rating is not None or args.max_rating is not None:
            suffix = f"_{args.min_rating or 'min'}to{args.max_rating or 'max'}"

        val_preds = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_preds)
        val_rmse = np.sqrt(val_mse)

        train_preds = model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_preds)
        train_rmse = np.sqrt(train_mse)

        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)

        status = "interrupted" if interrupted else "complete"
        print(f"\nTraining {status}.")
        print(f"  Train MSE: {train_mse:.2f} (RMSE: {train_rmse:.2f})")
        print(f"  Val   MSE: {val_mse:.2f} (RMSE: {val_rmse:.2f})")

        if args.model_type == "xgboost":
            mlflow.xgboost.log_model(model, f"model{suffix}")
        else:
            mlflow.lightgbm.log_model(model, f"model{suffix}Filtered")

        out_dir = "./results/filtered"
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([{
            'Model': args.model_type,
            'Min_Rating': args.min_rating,
            'Max_Rating': args.max_rating,
            'Interrupted': interrupted,
            'Validation_MSE': val_mse,
            'Validation_RMSE': val_rmse,
            'Train_MSE': train_mse,
            'Train_RMSE': train_rmse,
        }]).to_csv(f"{out_dir}/{args.model_type}_results{suffix}.csv", index=False)
