import numpy as np
import os
import argparse
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import mlflow

from dataset import load_data, build_features, encode_themes


def flatten_maia_embeddings(maia_seq):
    return maia_seq.mean(axis=1)


def prepare_features(X_struct, X_themes, maia_seq_flat, move_lengths, stockfish_features=None):
    lengths_2d = move_lengths.reshape(-1, 1).astype(np.float32)
    X = np.concatenate([X_struct, X_themes, lengths_2d], axis=1)
    if stockfish_features is not None:
        X = np.concatenate([X, stockfish_features], axis=1)
    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost on specific puzzle rating brackets.")
    parser.add_argument("--min_rating", type=int, default=None, help="Minimum rating (inclusive)")
    parser.add_argument("--max_rating", type=int, default=None, help="Maximum rating (exclusive)")
    args = parser.parse_args()
    
    csv_path = "./data/p200k.csv"
    # embeddings_path = None
    embeddings_path = "./data/p200k/maia2.npy"
    stockfish_path = "./data/p200k_sf_evals.csv"
    num_rows = None
    min_rating = args.min_rating
    max_rating = args.max_rating
    
    mlflow.set_experiment("Chess_Puzzle_Rating_Prediction")
    
    df, maia_seq, stockfish_features = load_data(csv_path, embeddings_path, stockfish_path, num_rows=num_rows)
    X_struct, move_lengths = build_features(df)
    X_themes = encode_themes(df)
    maia_seq_flat = flatten_maia_embeddings(maia_seq)
    y = df['Rating'].values

    mask = np.ones(len(y), dtype=bool)
    if min_rating is not None:
        mask &= (y >= min_rating)
    if max_rating is not None:
        mask &= (y < max_rating)
        
    X_struct = X_struct[mask]
    X_themes = X_themes[mask]
    maia_seq_flat = maia_seq_flat[mask]
    move_lengths = move_lengths[mask]
    if stockfish_features is not None:
        stockfish_features = stockfish_features[mask]
    y = y[mask]
    df = df[mask]
    max_num_rows = 60000
    df = df.head(max_num_rows)
    
    X = prepare_features(X_struct, X_themes, maia_seq_flat, move_lengths, stockfish_features)
    print(f"Total feature dimension: {X.shape[1]}")
    print(f"  Struct features: {X_struct.shape[1]}")
    print(f"  Theme features:  {X_themes.shape[1]}")
    print(f"  Maia embeddings: {maia_seq_flat.shape[1]}")
    print(f"  Length feature:   1")
    if stockfish_features is not None:
        print(f"  Stockfish features: {stockfish_features.shape[1]}")
    
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    xgb_params = {
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
    
    print(f"\nTraining XGBoost with {len(X_train)} samples, validating on {len(X_val)} samples")
    
    with mlflow.start_run():
        mlflow.log_param("min_rating", min_rating)
        mlflow.log_param("max_rating", max_rating)
        mlflow.log_param("model_type", "XGBoost_GPU")
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("num_train", len(X_train))
        mlflow.log_param("num_val", len(X_val))
        for k, v in xgb_params.items():
            mlflow.log_param(k, v)
        
        model = xgb.XGBRegressor(**xgb_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100,
        )
        
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
        
        print(f"\nTraining complete.")
        print(f"  Train MSE: {train_mse:.2f} (RMSE: {train_rmse:.2f})")
        print(f"  Val   MSE: {val_mse:.2f} (RMSE: {val_rmse:.2f})")
        print(f"  Best iteration: {model.best_iteration}")
        
        out_dir = "./results/p200k"
        os.makedirs(out_dir, exist_ok=True)
        result = pd.DataFrame([{
            'Min_Rating': min_rating,
            'Max_Rating': max_rating,
            'Validation_MSE': val_mse,
            'Validation_RMSE': val_rmse,
            'Train_MSE': train_mse,
            'Train_RMSE': train_rmse,
            'Best_Iteration': model.best_iteration,
        }])
        
        suffix = ""
        if min_rating is not None or max_rating is not None:
            suffix = f"_{min_rating or 'min'}to{max_rating or 'max'}"

        mlflow.xgboost.log_model(model, f"xgboost_baseline_model{suffix}")    
            
        result.to_csv(f"{out_dir}/xgboost_baseline_results_with_stockfishDiff{suffix}.csv", index=False)
