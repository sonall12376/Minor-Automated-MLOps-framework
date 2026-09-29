from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "india_weather_rainfall_data.xlsx"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


def load_dataset():

    return pd.read_excel(DATA_PATH)


def preprocess(df):

    df = df.copy()

    df["date_of_record"] = pd.to_datetime(
        df["date_of_record"]
    )

    df = df.sort_values(
        ["station_name", "date_of_record"]
    )

    df["RainTomorrow"] = (
        df.groupby("station_name")["rainfall"]
        .shift(-1)
    )

    df["RainTomorrow"] = (
        df["RainTomorrow"] >= 1.0
    ).astype(int)

    df = df.dropna().reset_index(drop=True)

    return df


def save_dataset(df):

    output_path = (
        OUTPUT_PATH
        / "weather_preprocessed.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    return output_path


if __name__ == "__main__":

    dataset = load_dataset()

    processed_dataset = preprocess(
        dataset
    )

    path = save_dataset(
        processed_dataset
    )

    print()

    print("===================================")
    print("Preprocessing Completed")
    print("===================================")

    print(path)

    print()

    print(processed_dataset.head())

    print()

    print(processed_dataset.shape)