import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    df = df.dropna()
    return df

def transform_dates(df):
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"])
        df["Year"] = df["Order Date"].dt.year
        df["Month"] = df["Order Date"].dt.month
    return df

def encode_categoricals(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    return df

def prepare_features(df, target_column="Sales"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

