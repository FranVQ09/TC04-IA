import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    df = df.dropna()
    return df

def transform_dates(df):
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Order Date"]) 
        df["Year"] = df["Order Date"].dt.year
        df["Month"] = df["Order Date"].dt.month
    return df

def encode_subcategory_column(df):
    le = LabelEncoder()
    df["Sub-Category"] = le.fit_transform(df["Sub-Category"])
    return df, le

def encode_categoricals(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    return df

def drop_unnecessary_columns(df):
    cols_to_drop = [
        "Order ID", "Ship Date", "Product ID", "Product Name", "Ship Mode"
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    return df 

def prepare_features(df, target_column="Sales"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def aggregate_monthly_sales(df):
    grouped = (
        df.groupby(["Sub-Category", "Year", "Month"])
        .agg({"Sales": "sum"})
        .reset_index()
        .sort_values(by=["Sub-Category", "Year", "Month"])
    )
    return grouped

def generate_2025_input(monthly_data):
    subcats = monthly_data["Sub-Category"].unique()
    months = list(range(1, 13))
    year = 2025

    future_rows = []

    for subcat in subcats:
        for month in months:
            future_rows.append({
                "Sub-Category": subcat,
                "Year": year,
                "Month": month,
            })
    
    return pd.DataFrame(future_rows)
