import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dataset import build_features, encode_themes

def prepare_features(X_struct, X_themes, move_lengths, stockfish_features=None):
    lengths_2d = move_lengths.reshape(-1, 1).astype(np.float32)
    X = np.concatenate([X_struct, X_themes, lengths_2d], axis=1)
    if stockfish_features is not None:
        X = np.concatenate([X, stockfish_features], axis=1)
    return X

def load_best_model_for_range(runs, min_rating, max_rating, suffix):
    if min_rating is None:
        mask = runs['params.min_rating'].isnull() | (runs['params.min_rating'] == 'None')
    else:
        mask = runs['params.min_rating'] == str(min_rating)
        
    if max_rating is None:
        mask &= runs['params.max_rating'].isnull() | (runs['params.max_rating'] == 'None')
    else:
        mask &= runs['params.max_rating'] == str(max_rating)
        
    matching_runs = runs[mask]
    if len(matching_runs) == 0:
        raise ValueError(f"No models found for range {min_rating} to {max_rating}")
        
    best_run = matching_runs.sort_values("metrics.val_rmse", ascending=True).iloc[0]
    return mlflow.xgboost.load_model(f"runs:/{best_run['run_id']}/xgboost_baseline_model{suffix}")

def get_base_predictions(models, X):
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict(X)
    return preds
    
def create_meta_features(predictions, base_features):
    pred_stack = np.column_stack([
        predictions['model_low'],
        predictions['model_mid'],
        predictions['model_high']
    ])
    
    pred_mean = np.mean(pred_stack, axis=1).reshape(-1, 1)
    pred_std = np.std(pred_stack, axis=1).reshape(-1, 1)
    pred_max = np.max(pred_stack, axis=1).reshape(-1, 1)
    pred_min = np.min(pred_stack, axis=1).reshape(-1, 1)
    
    diff_mid_low = np.abs(predictions['model_mid'] - predictions['model_low']).reshape(-1, 1)
    diff_high_mid = np.abs(predictions['model_high'] - predictions['model_mid']).reshape(-1, 1)
    diff_high_low = np.abs(predictions['model_high'] - predictions['model_low']).reshape(-1, 1)
    
    return np.concatenate([
        base_features, 
        pred_stack, 
        pred_mean, pred_std, pred_max, pred_min,
        diff_mid_low, diff_high_mid, diff_high_low
    ], axis=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rows", type=int, default=60000)
    args = parser.parse_args()
    
    df_train_ids = set(pd.read_csv("./data/p200k.csv", usecols=["PuzzleId"])["PuzzleId"])
    df_holdout = pd.read_csv("../filtered.csv")
    df_holdout = df_holdout[~df_holdout["PuzzleId"].isin(df_train_ids)].copy()
    df_holdout = df_holdout.head(args.num_rows)
    
    y = df_holdout['Rating'].values
    
    X_struct, move_lengths = build_features(df_holdout)
    X_themes = encode_themes(df_holdout)
    stockfish_features = './data/p200k_sf_evals.csv'  
    
    X_base = prepare_features(X_struct, X_themes, move_lengths, stockfish_features)
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db") 
    runs = mlflow.search_runs(experiment_names=["Chess_Puzzle_Rating_Prediction"])
    
    models = {
        'model_low': load_best_model_for_range(runs, None, 1200, "_minto1200"),
        'model_mid': load_best_model_for_range(runs, 1200, 1800, "_1200to1800"),
        'model_high': load_best_model_for_range(runs, 1800, None, "_1800tomax")
    }
    
    predictions = get_base_predictions(models, X_base)
    X_meta = create_meta_features(predictions, X_base)
    
    X_train, X_val, y_train, y_val = train_test_split(X_meta, y, test_size=0.1, random_state=42)
    
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    
    catboost_params = {
        'iterations': 5000,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 100,
        'task_type': 'CPU',
        'verbose': 200,
        'random_seed': 42
    }
    
    meta_model = CatBoostRegressor(**catboost_params)
    meta_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    val_preds = meta_model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_preds)
    val_rmse = np.sqrt(val_mse)
    
    meta_model.save_model("catboost_meta_ensemble.cbm")
    
    out_dir = "./results/ensamble"
    os.makedirs(out_dir, exist_ok=True)
    
    results_df = pd.DataFrame([{
        'Holdout_Size': len(df_holdout),
        'Num_Rows_Requested': args.num_rows,
        'CatBoost_Iterations': catboost_params['iterations'],
        'CatBoost_Depth': catboost_params['depth'],
        'Validation_MSE': val_mse,
        'Validation_RMSE': val_rmse,
        'Best_Iteration': meta_model.get_best_iteration()
    }])
    
    csv_path = f"{out_dir}/catboost_results_holdout_{len(df_holdout)}rows.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"MSE: {val_mse:.2f} | RMSE: {val_rmse:.2f} | Saved to: {csv_path}")
