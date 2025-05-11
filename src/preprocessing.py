import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    df = df.dropna()
    return df

def transform_dates(df):
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
        df["Year"] = df["Order Date"].dt.year
        df["Month"] = df["Order Date"].dt.month
        df = df.drop(columns=["Order Date"])
    return df

def drop_unused_columns(df):
    cols_to_drop = [
        "Row ID", "Order ID", "Customer ID", "Customer Name",
        "Product ID", "Product Name", "Postal Code", "Ship Date"
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")
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

