from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_preprocessed.csv"
)


def load_data():

    return pd.read_csv(DATA_PATH)


if __name__ == "__main__":

    df = load_data()

    print("\n========== DATASET SUMMARY ==========\n")

    print(df.info())

    print("\n========== MISSING VALUES ==========\n")

    print(df.isnull().sum())

    print("\n========== DUPLICATE ROWS ==========\n")

    print(df.duplicated().sum())

    print("\n========== TARGET DISTRIBUTION ==========\n")

    print(df["RainTomorrow"].value_counts())

    print("\n========== DESCRIPTIVE STATISTICS ==========\n")

    print(df.describe())