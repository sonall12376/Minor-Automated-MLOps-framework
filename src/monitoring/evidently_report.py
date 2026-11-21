from pathlib import Path

import json
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_PATH = (
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

REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "evidently"
)

REPORT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


def load_baseline():

    return pd.read_csv(BASELINE_PATH)


def load_latest_batch():

    files = sorted(
        PRODUCTION_PATH.glob("*.csv")
    )

    return pd.read_csv(files[-1])


def generate_report(baseline, production):

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    # Use the same features as our custom PSI/KS detector
    
    numerical_columns = baseline.select_dtypes(
    include=["int64", "float64"]
    ).columns
    
    numerical_columns = [
        col for col in numerical_columns
        if col != "RainTomorrow"
]
    reference_features = baseline[numerical_columns].copy()
    current_features = production[numerical_columns].copy()
    
    snapshot = report.run(
        reference_data=reference_features,
        current_data=current_features
)

    html_path = (
        REPORT_PATH
        / "drift_report.html"
    )

    json_path = (
        REPORT_PATH
        / "evidently_results.json"
    )

    # Save HTML report
    snapshot.save_html(
        str(html_path)
    )

    # Extract machine-readable results
    results = snapshot.dict()

    # Save JSON report
    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            default=str
        )

    return html_path

