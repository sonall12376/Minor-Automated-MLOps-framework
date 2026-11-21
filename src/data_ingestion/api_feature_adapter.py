
from pathlib import Path
import json
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "weather_preprocessed.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models"
    / "baseline_model.pkl"
)

API_DIR = PROJECT_ROOT / "data" / "api_staging"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "api_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = {
    1: "Winter", 2: "Winter", 3: "Winter",
    4: "Summer", 5: "Summer", 6: "Summer",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
    12: "Winter"
}


def prepare_api_features():

    files = sorted(API_DIR.glob("open_meteo_multi_*.csv"))

    if not files:
        raise FileNotFoundError("No API data found.")

    api = pd.read_csv(files[-1])
    historical = pd.read_csv(HISTORICAL_PATH)

    model = joblib.load(MODEL_PATH)
    feature_names = list(model.feature_names_in_)

    # Build station metadata from historical data.
    metadata = (
        historical.groupby("station_name")[["state", "district"]]
        .first()
        .reset_index()
    )

    api = api.merge(
        metadata,
        on="station_name",
        how="left",
        validate="many_to_one"
    )

    if api[["state", "district"]].isna().any().any():
        raise ValueError("Missing station metadata.")

    api["date_of_record"] = pd.to_datetime(
        api["date_of_record"]
    )

    api["month"] = api["date_of_record"].dt.month_name()
    api["season"] = api["date_of_record"].dt.month.map(SEASONS)

    api["day"] = api["date_of_record"].dt.day
    api["month_num"] = api["date_of_record"].dt.month
    api["year"] = api["date_of_record"].dt.year

    categorical = pd.get_dummies(
        api[["month", "season", "state", "district"]],
        drop_first=False,
        dtype=int
    )

    # Reconstruct the training-time drop_first encoding.
    historical_categories = {
        col: sorted(historical[col].dropna().unique())
        for col in ["month", "season", "state", "district"]
    }

    for col, categories in historical_categories.items():
        if categories:
            first_category = categories[0]
            dummy_column = f"{col}_{first_category}"
            categorical = categorical.drop(
                columns=[dummy_column],
                errors="ignore"
            )

    # Only prepare features whose meaning is currently verified.
    verified_numeric = [
        "avg_temp", "min_temp", "max_temp",
        "latitude", "longitude",
        "day", "month_num", "year"
    ]

    X_partial = pd.concat(
        [
            api[verified_numeric].reset_index(drop=True),
            categorical.reset_index(drop=True)
        ],
        axis=1
    )

    missing_features = sorted(
        set(feature_names) - set(X_partial.columns)
    )

    # Save an alignment preview, not model-ready input.
    X_preview = X_partial.reindex(
        columns=feature_names
    )

    preview_path = OUTPUT_DIR / "api_feature_preview.csv"
    X_preview.to_csv(preview_path, index=False)

    report = {
        "api_records": len(api),
        "expected_model_features": len(feature_names),
        "prepared_columns": len(X_partial.columns),
        "missing_model_features": missing_features,
        "model_ready": False,
        "note": (
            "Numerical feature compatibility is incomplete. "
            "Do not use this preview for model prediction or SHAP."
        )
    }

    report_path = OUTPUT_DIR / "feature_alignment_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\nAPI Feature Adapter Completed")
    print("API records:", len(api))
    print("Expected model features:", len(feature_names))
    print("Missing model features:", missing_features)
    print("Model ready:", False)
    print("Report:", report_path)


if __name__ == "__main__":
    prepare_api_features()
