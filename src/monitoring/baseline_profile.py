from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_preprocessed.csv"
)

PROFILE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "profiles"
)

PROFILE_PATH.mkdir(
    parents=True,
    exist_ok=True
)


def load_dataset():

    return pd.read_csv(DATA_PATH)


def create_baseline_profile(df):

    profile = {}

    profile["dataset_shape"] = {

        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])

    }

    profile["feature_names"] = df.columns.tolist()

    profile["target_distribution"] = (

        df["RainTomorrow"]
        .value_counts(normalize=True)
        .to_dict()

    )

    numerical_columns = df.select_dtypes(

        include=["int64", "float64"]

    ).columns

    profile["numerical_features"] = {}

    for column in numerical_columns:

        profile["numerical_features"][column] = {

            "mean": float(df[column].mean()),
            "std": float(df[column].std()),
            "median": float(df[column].median()),
            "min": float(df[column].min()),
            "max": float(df[column].max())

        }

    categorical_columns = df.select_dtypes(

        include=["object"]

    ).columns

    profile["categorical_features"] = {}

    for column in categorical_columns:

        profile["categorical_features"][column] = (

            df[column]
            .value_counts(normalize=True)
            .to_dict()

        )

    return profile


def save_profile(profile):

    output_path = (

        PROFILE_PATH
        / "baseline_profile.json"

    )

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(

            profile,

            file,

            indent=4

        )

    return output_path


if __name__ == "__main__":

    dataset = load_dataset()

    baseline_profile = create_baseline_profile(
        dataset
    )

    saved_path = save_profile(
        baseline_profile
    )

    print()

    print("==============================")

    print("Baseline Profile Created")

    print("==============================")

    print(saved_path)