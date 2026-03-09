import pandas as pd
import numpy as np
import os
import chess
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

def load_data(csv_path, embeddings_path, num_rows=None):
    df = pd.read_csv(csv_path)
    if num_rows:
        df = df.head(num_rows)
    maia_embeddings = np.load(embeddings_path)
    if len(maia_embeddings) != len(df):
        maia_embeddings = maia_embeddings[:len(df)]
    return df, maia_embeddings

def extract_board_stats(fen):
    board = chess.Board(fen)
    counts = {
        'white_pieces': int(board.occupied_co[chess.WHITE]).bit_count(),
        'black_pieces': int(board.occupied_co[chess.BLACK]).bit_count(),
        'material_balance': sum([len(board.pieces(piece, chess.WHITE)) * val 
                               for piece, val in zip([1,2,3,4,5,6], [1,3,3,5,9,0])]) - \
                           sum([len(board.pieces(piece, chess.BLACK)) * val 
                               for piece, val in zip([1,2,3,4,5,6], [1,3,3,5,9,0])])
    }
    return counts

def build_features(df):
    feats = pd.DataFrame(index=df.index)
    feats['SolutionLength'] = df['Moves'].apply(lambda x: len(str(x).split()))
    feats['IsWhiteToMove'] = df['FEN'].apply(lambda x: 1 if x.split()[1] == 'w' else 0)
    stats = df['FEN'].apply(extract_board_stats).apply(pd.Series)
    feats = pd.concat([feats, stats], axis=1)
    prob_cols = [c for c in df.columns if 'success_prob_blitz' in c]
    feats = pd.concat([feats, df[prob_cols]], axis=1)
    return feats

def encode_themes(df):
    themes_list = df['Themes'].apply(lambda x: x.split())
    mlb = MultiLabelBinarizer()
    themes_encoded = mlb.fit_transform(themes_list)
    themes_df = pd.DataFrame(themes_encoded, columns=mlb.classes_, index=df.index)
    return themes_df

def run_pipeline(csv_path, embeddings_path, num_rows=None):
    df, maia_3d = load_data(csv_path, embeddings_path, num_rows=num_rows)
    X_struct = build_features(df)
    X_themes = encode_themes(df)
    X_maia = maia_3d.reshape(len(maia_3d), -1)
    X = np.hstack([X_struct.values, X_themes.values, X_maia])
    y = df['Rating'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mses = []
    results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'device': 'cpu'
        }
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        mses.append(mse)
        results.append({'Fold': fold + 1, 'MSE': mse, 'RMSE': rmse})
        print(f"Fold {fold+1} MSE: {mse:.2f}, RMSE: {rmse:.2f}")
    
    avg_mse = np.mean(mses)
    avg_rmse = np.sqrt(avg_mse)
    print(f"Final Average RMSE: {avg_rmse:.2f}")
    
    results.append({'Fold': 'Final Average', 'MSE': avg_mse, 'RMSE': avg_rmse})
    results_df = pd.DataFrame(results)
    results_df.to_csv("../features/model_results.csv", index=False)
    print("Results saved to ../features/model_results.csv")

if __name__ == "__main__":
    run_pipeline("./data/puzzle50000.csv", "./latentFeatures/maia2_latent_subset_2k.npy", num_rows=1000)
