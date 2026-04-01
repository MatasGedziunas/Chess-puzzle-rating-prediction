from dataset.loaders import load_data
from dataset.board_features import build_features, encode_themes, build_advanced_features, build_success_prob_features
import os

csv_path = "../filtered.csv"
df, _, _ = load_data(
        csv_path,
        None,
        None,
        False,
)

data_file_name = os.path.splitext(os.path.basename(csv_path))[0]
advanced_features = build_advanced_features(df, data_file_name)
