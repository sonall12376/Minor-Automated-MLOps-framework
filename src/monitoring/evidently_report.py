from pathlib import Path

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


def generate_report(
    baseline,
    production
):

    report = Report(

        metrics=[
            DataDriftPreset()
        ]

    )

    snapshot = report.run(
        reference_data=baseline,
        current_data=production

)
    output_path = (
        REPORT_PATH
        / "drift_report.html"
)
    snapshot.save_html(
        str(output_path)
)
    return output_path


