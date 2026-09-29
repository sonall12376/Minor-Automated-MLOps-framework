from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_preprocessed.csv"
)

PRODUCTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "production"
)

PRODUCTION_PATH.mkdir(
    parents=True,
    exist_ok=True
)


def load_dataset():

    return pd.read_csv(DATA_PATH)


def create_production_batch(df, batch_size=5000):

    production_batch = df.sample(
        n=batch_size,
        random_state=42
    ).reset_index(drop=True)

    return production_batch


def save_batch(batch):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    output_path = (
        PRODUCTION_PATH
        / f"production_{timestamp}.csv"
    )

    batch.to_csv(
        output_path,
        index=False
    )

    return output_path


if __name__ == "__main__":

    dataset = load_dataset()

    production_batch = create_production_batch(
        dataset
    )

    saved_path = save_batch(
        production_batch
    )

    print()

    print("==============================")
    print("Production Batch Created")
    print("==============================")

    print(saved_path)

    print()

    print(production_batch.shape)