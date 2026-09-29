from pathlib import Path
import json

import pandas as pd


from drift_metrics import (
    calculate_psi,
    calculate_ks_test,
    get_drift_severity,
    get_recommendation
)

from evidently_report import generate_report



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
    / "drift_reports"
)

REPORT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


def load_baseline():

    return pd.read_csv(
        BASELINE_PATH
    )


def load_latest_batch():

    files = sorted(
        PRODUCTION_PATH.glob("*.csv")
    )

    return pd.read_csv(
        files[-1]
    )


def detect_drift(
    baseline,
    production
):

    feature_report = {}

    numerical_columns = baseline.select_dtypes(
        include=["int64", "float64"]
    ).columns

    numerical_columns = [
        column
        for column in numerical_columns
        if column != "RainTomorrow"
    ]

    for column in numerical_columns:

        psi = calculate_psi(
            baseline[column],
            production[column]
        )

        ks_statistic, ks_pvalue = calculate_ks_test(
            baseline[column],
            production[column]
        )

        severity = get_drift_severity(
            psi
        )

        recommendation = get_recommendation(
            psi,
            ks_pvalue
        )

        feature_report[column] = {

            "psi": round(float(psi), 4),

            "ks_statistic": round(
                float(ks_statistic),
                4
            ),

            "ks_pvalue": round(
                float(ks_pvalue),
                4
            ),

            "drift_status": severity,

            "drift_detected": (
                severity != "No Drift"
            ),

            "recommendation": recommendation

        }

    drifted_features = sum(

        1

        for feature in feature_report.values()

        if feature["drift_detected"]

    )

    total_features = len(feature_report)

    health_score = round(

        (
            (total_features - drifted_features)

            / total_features
        ) * 100,

        2

    )

    if drifted_features == 0:

        overall_status = "Healthy"

    elif drifted_features <= 2:

        overall_status = "Low Drift"

    elif drifted_features <= 5:

        overall_status = "Moderate Drift"

    else:

        overall_status = "High Drift"

    report = {

        "summary": {

            "overall_status": overall_status,

            "health_score": health_score,

            "total_features": total_features,

            "drifted_features": drifted_features

        },

        "feature_drift": feature_report

    }

    return report


def save_report(report):

    output_path = (

        REPORT_PATH
        / "drift_report.json"

    )

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    return output_path


if __name__ == "__main__":

    baseline = load_baseline()

    production = load_latest_batch()

    drift_report = detect_drift(

        baseline,

        production

    )

    # print(json.dumps(drift_report, indent=4))

    evidently_report_path = generate_report(
    baseline,
    production
)
    saved_path = save_report(
        drift_report
)
    print()
    print("==============================")
    print("Drift Detection Completed")
    print("==============================")
    print(f"Statistical Report : {saved_path}")
    print(f"Evidently Report   : {evidently_report_path}")