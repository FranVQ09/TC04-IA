import pandas as pd

def load_csv(path="data/sales-forecasting/train.csv"):
    return pd.read_csv(path)
