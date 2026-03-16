import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import mlflow

from dataset import load_data, build_features, encode_themes


def flatten_maia_embeddings(maia_seq):
    return maia_seq.mean(axis=1)


def prepare_features(X_struct, X_themes, maia_seq_flat, move_lengths):
    lengths_2d = move_lengths.reshape(-1, 1).astype(np.float32)
    X = np.concatenate([X_struct, X_themes, maia_seq_flat, lengths_2d], axis=1)
    return X


if __name__ == "__main__":
    
    csv_path = "./data/p200k.csv"
    embeddings_path = None
    # embeddings_path = "../features/maia2_sequence_embeddings.npy"
    num_rows = None
    
    mlflow.set_experiment("Chess_Puzzle_Rating_Prediction")
    
    df, maia_seq = load_data(csv_path, embeddings_path, num_rows=num_rows)
    X_struct, move_lengths = build_features(df)
    X_themes = encode_themes(df)
    maia_seq_flat = flatten_maia_embeddings(maia_seq)
    y = df['Rating'].values
    
    X = prepare_features(X_struct, X_themes, maia_seq_flat, move_lengths)
    print(f"Total feature dimension: {X.shape[1]}")
    print(f"  Struct features: {X_struct.shape[1]}")
    print(f"  Theme features:  {X_themes.shape[1]}")
    print(f"  Maia embeddings: {maia_seq_flat.shape[1]}")
    print(f"  Length feature:   1")
    
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    xgb_params = {
        "n_estimators": 5000,
        "learning_rate": 0.1,
        "max_leaves": 63,
        "max_bin": 63,
        "tree_method": "gpu_hist",
        "device": "cuda",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "verbosity": 1,
    }
    
    print(f"\nTraining XGBoost with {len(X_train)} samples, validating on {len(X_val)} samples")
    
    with mlflow.start_run():
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
        
        mlflow.xgboost.log_model(model, "xgboost_baseline_model")
        
        out_dir = "./results/p200k"
        os.makedirs(out_dir, exist_ok=True)
        result = pd.DataFrame([{
            'Validation_MSE': val_mse,
            'Validation_RMSE': val_rmse,
            'Train_MSE': train_mse,
            'Train_RMSE': train_rmse,
            'Best_Iteration': model.best_iteration,
        }])
        result.to_csv(f"{out_dir}/xgboost_baseline_results.csv", index=False)
