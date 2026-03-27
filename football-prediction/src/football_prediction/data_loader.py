from pathlib import Path
import pandas as pd


def load_and_merge_data(data_path: str) -> pd.DataFrame:
    files = list(Path(data_path).glob("*.csv"))
    dfs = [pd.read_csv(file) for file in files]
    return pd.concat(dfs, ignore_index=True)