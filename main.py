from src.load_data import load_csv
from src.eda import explore_data
from src.preprocessing import clean_data, transform_dates, encode_categoricals, prepare_features
from src.modeling import train_and_evaluate, train_with_gridsearch, train_rf_with_gridsearch, train_dt_with_gridsearch

def main():
    df = load_csv()
    explore_data(df)
    df = clean_data(df)
    df = transform_dates(df)
    df = encode_categoricals(df)
    X, y = prepare_features(df, target_column="Sales")
    train_and_evaluate(X, y)
    train_rf_with_gridsearch(X, y)
    train_with_gridsearch(X, y)
    train_dt_with_gridsearch(X, y)


if __name__ == "__main__":
    main()
