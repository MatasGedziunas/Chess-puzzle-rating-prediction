import os

from dataset.board_features import build_features
from dataset.loaders import load_data

csv_path = "../filtered.csv"
df, _, _ = load_data(
        csv_path,
        None,
        None,
        False,
)

data_file_name = os.path.splitext(os.path.basename(csv_path))[0]
build_features(df, save_csv_path=f"./data/{data_file_name}_struct_features.csv")
