from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "india_weather_rainfall_data.xlsx"
)


def load_data():

    df = pd.read_excel(DATA_PATH)

    return df


if __name__ == "__main__":

    dataset = load_data()

    print("\nDataset Loaded Successfully\n")

    print("Shape :", dataset.shape)

    print("\nColumns:\n")

    print(dataset.columns.tolist())

    print("\nFirst 5 Rows:\n")

    print(dataset.head())