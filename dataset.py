import pandas as pd
import numpy as np
import os
import chess
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
from maia2 import model, inference
from tqdm import tqdm


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.latent_vector = None
        self.hook = self.model.last_ln.register_forward_hook(self.save_output)

    def save_output(self, module, input, output):
        self.latent_vector = output

    def __call__(self, boards, elo_self, elo_oppo):
        self.model(boards, elos_self=elo_self, elos_oppo=elo_oppo)
        return self.latent_vector


def get_stockfish_features(row):
    STOCKFISH_PATH = "./stockfish" 
    puzzle_id, fen = row['PuzzleId'], row['FEN']
    
    process = subprocess.Popen(
        [STOCKFISH_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    commands = f"position fen {fen}\neval\nquit\n"
    output, _ = process.communicate(commands)
    
    metrics = {
        "PuzzleId": puzzle_id,
        "SF_Material": None,
        "SF_Positional": None,
        "SF_Final_Eval": None
    }
    
    if "Final evaluation: none (in check)" in output:
        process2 = subprocess.Popen(
            [STOCKFISH_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process2.stdin.write(f"position fen {fen}\ngo depth 10\n")
        process2.stdin.flush()
        
        for line in process2.stdout:
            line = line.strip()
            if 'score cp' in line:
                match = re.search(r'score cp (-?\d+)', line)
                if match:
                    metrics["SF_Final_Eval"] = int(match.group(1)) / 100.0
            elif 'score mate' in line:
                match = re.search(r'score mate (-?\d+)', line)
                if match:
                    mate_in = int(match.group(1))
                    metrics["SF_Final_Eval"] = 100.0 if mate_in > 0 else -100.0
            elif line.startswith('bestmove'):
                break

        
        process2.stdin.write("quit\n")
        process2.stdin.flush()
        process2.wait()

    else:
        # Normal parsing for non-check positions
        lines = output.split('\n')
        for line in lines:
            if "<-- this bucket is used" in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 5:
                    try:
                        metrics["SF_Material"] = float(parts[2].replace(" ", ""))
                        metrics["SF_Positional"] = float(parts[3].replace(" ", ""))
                    except ValueError:
                        pass
                        
            elif "Final evaluation" in line and "(white side)" in line:
                match = re.search(r"([+-]?\d+\.\d+)", line)
                if match:
                    metrics["SF_Final_Eval"] = float(match.group(1))
                    
    return metrics

    
def process_all_puzzles(input_csv, output_csv, max_workers=8):
    df = pd.read_csv(input_csv)
    
    already_processed = set()
    existing_results = []
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        done_mask = (
            # existing_df['SF_Material'].notna() &
            # existing_df['SF_Positional'].notna() &
            existing_df['SF_Final_Eval'].notna()
        )
        done_df = existing_df[done_mask]
        already_processed = set(done_df['PuzzleId'].tolist())
        existing_results = done_df.to_dict('records')
        print(f"Loaded {len(already_processed)} already-processed puzzles from {output_csv}")
    
    tasks = df[~df['PuzzleId'].isin(already_processed)][['PuzzleId', 'FEN']].to_dict('records')
    print(f"Remaining puzzles to process: {len(tasks)}")
    
    if len(tasks) == 0:
        print("All puzzles already processed.")
        return
    
    new_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_stockfish_features, row): row for row in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Puzzles"):
            try:
                res = future.result()
                new_results.append(res)
            except Exception as e:
                print(f"Error processing puzzle: {e}")
    
    all_results = existing_results + new_results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved {len(all_results)} total results to {output_csv}")



def process_puzzle_sequences(df, elo_indices=[0, 5, 10], max_steps=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    maia2_model = model.from_pretrained(type="rapid", device=device)
    maia2_model.to(device)
    maia2_model.eval()
    
    extractor = FeatureExtractor(maia2_model)

    num_puzzles = len(df)
    num_elos = len(elo_indices)
    sequence_results = np.zeros((num_puzzles, max_steps, num_elos, 1024), dtype=np.float32)
    
    for p_idx, row in tqdm(df.iterrows(), total=num_puzzles, desc="Computing Maia2 Embeddings"):
        board = chess.Board(row['FEN'])
        moves = str(row['Moves']).split()
        
        puzzle_fens = [board.fen()]
        for move_str in moves[:max_steps-1]:
            try:
                move = chess.Move.from_uci(move_str)
                board.push(move)
                puzzle_fens.append(board.fen())
            except:
                break
        
        num_steps = len(puzzle_fens)
        batch_tensors = [inference.board_to_tensor(chess.Board(f)) for f in puzzle_fens]
        boards = torch.stack(batch_tensors).to(device)
        
        for elo_idx_in_list, skill_val in enumerate(elo_indices):
            elos = torch.full((num_steps,), skill_val, device=device).long()
            with torch.no_grad():
                latent = extractor(boards, elos, elos)
                if len(latent.shape) > 2:
                    latent = latent.mean(dim=1)  
                sequence_results[p_idx, :num_steps, elo_idx_in_list, :] = latent.cpu().numpy()
                
    sequence_results = sequence_results.reshape(num_puzzles, max_steps, -1)
    return sequence_results


def load_data(csv_path, embeddings_path=None, stockfish_path=None, num_rows=None):
    df = pd.read_csv(csv_path)
    if num_rows:
        df = df.head(num_rows)
        
    if embeddings_path and os.path.exists(embeddings_path):
        maia_embeddings = np.load(embeddings_path)
        if len(maia_embeddings) != len(df):
            raise ValueError("Embeddings length does not match dataset length")
    else:
        maia_embeddings = process_puzzle_sequences(df)
        
        dataset_name = Path(csv_path).stem
        save_dir = f"./data/{dataset_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/maia2.npy"
        np.save(save_path, maia_embeddings)
        print(f"Saved computed embeddings to {save_path}")
        
    if stockfish_path:
        sf_df = pd.read_csv(stockfish_path)
        df = df.merge(sf_df, on='PuzzleId', how='left')
        sf_cols = ['SF_Material', 'SF_Positional', 'SF_Final_Eval']
        stockfish_features = df[sf_cols].fillna(0).values.astype(np.float32)
    else: 
        stockfish_features = None
    
    return df, maia_embeddings, stockfish_features


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
    
    length = feats.pop('SolutionLength').values
    return feats.values.astype(np.float32), length


def encode_themes(df):
    themes_list = df['Themes'].apply(lambda x: x.split())
    mlb = MultiLabelBinarizer()
    themes_encoded = mlb.fit_transform(themes_list)
    return themes_encoded.astype(np.float32)

    


class ChessPuzzleDataset(Dataset):
    def __init__(self, X_struct, X_themes, X_maia_seq, move_lengths, ratings):
        self.X_struct = torch.tensor(X_struct, dtype=torch.float32)
        self.X_themes = torch.tensor(X_themes, dtype=torch.float32)
        self.X_seq = torch.tensor(X_maia_seq, dtype=torch.float32)
        self.lengths = torch.tensor(np.clip(move_lengths, 0, 15), dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return (self.X_struct[idx], 
                self.X_themes[idx], 
                self.X_seq[idx], 
                self.lengths[idx], 
                self.ratings[idx])
