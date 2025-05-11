from src.load_data import load_csv
from src.eda import explore_data
from src.preprocessing import clean_data, transform_dates, encode_categoricals, prepare_features, drop_unused_columns
from src.modeling import train_and_evaluate

def main():
    df = load_csv()
    explore_data(df)
    df = clean_data(df)
    df = transform_dates(df)
    df = drop_unused_columns(df)
    df = encode_categoricals(df)
    X, y = prepare_features(df, target_column="Sales")
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()
