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
from decision_engine import make_decision



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

def load_evidently_results():

    json_path = (
        PROJECT_ROOT
        / "artifacts"
        / "evidently"
        / "evidently_results.json"
    )

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    for metric in data.get("metrics", []):

        if metric.get("metric_name", "").startswith(
            "DriftedColumnsCount("
        ):
            value = metric["value"]

            drifted_count = int(value["count"])
            drift_share = float(value["share"])

            if not 0 <= drift_share <= 1:
                raise ValueError("Invalid Evidently drift share")

            threshold = float(
                metric.get("config", {}).get(
                    "drift_share", 0.5
                )
            )

            return {
                "drifted_features": drifted_count,
                "drift_percentage": round(
                    drift_share * 100, 2
                ),
                "dataset_drift": drift_share >= threshold
            }

    raise ValueError(
        "DriftedColumnsCount metric not found in Evidently JSON"
    )


def detect_drift(baseline, production):

    feature_report = {}

    numerical_columns = baseline.select_dtypes(
        include=["int64", "float64"]
    ).columns

    numerical_columns = [
        column
        for column in numerical_columns
        if column != "RainTomorrow"
    ]

    if not numerical_columns:
        raise ValueError("No numerical features found")

    for column in numerical_columns:

        psi = calculate_psi(
            baseline[column],
            production[column]
        )

        ks_statistic, ks_pvalue = calculate_ks_test(
            baseline[column],
            production[column]
        )

        severity = get_drift_severity(psi)

        feature_report[column] = {
            "psi": round(float(psi), 4),
            "ks_statistic": round(
                float(ks_statistic), 4
            ),
            "ks_pvalue": round(
                float(ks_pvalue), 4
            ),
            "drift_status": severity,
            "drift_detected": severity != "No Drift",
            "recommendation": get_recommendation(
                psi, ks_pvalue
            )
        }

    total_features = len(feature_report)

    drifted_features = sum(
        feature["drift_detected"]
        for feature in feature_report.values()
    )

    health_score = round(
        100 * (total_features - drifted_features)
        / total_features,
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

    average_psi = round(
        sum(
            feature["psi"]
            for feature in feature_report.values()
        ) / total_features,
        4
    )

    minimum_pvalue = min(
        feature["ks_pvalue"]
        for feature in feature_report.values()
    )

    # Actual Evidently results
    evidently = load_evidently_results()

    decision = make_decision(
        overall_status,
        minimum_pvalue,
        evidently["dataset_drift"],
        evidently["drift_percentage"]
    )

    return {
        "summary": {
            "overall_status": overall_status,
            "health_score": health_score,
            "total_features": total_features,
            "drifted_features": drifted_features,
            "average_psi": average_psi,
            "minimum_ks_pvalue": minimum_pvalue,
            "evidently_dataset_drift": evidently[
                "dataset_drift"
            ],
            "evidently_drift_percentage": evidently[
                "drift_percentage"
            ],
            "evidently_drifted_features": evidently[
                "drifted_features"
            ],
            "decision": decision["decision"],
            "confidence": decision["confidence"],
            "recommended_action": decision["action"]
        },
        "feature_drift": feature_report
    }

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

    # Generate fresh Evidently results first
    evidently_report_path = generate_report(
        baseline,
        production
    )

    # Then combine custom metrics and Evidently
    drift_report = detect_drift(
        baseline,
        production
    )

    saved_path = save_report(drift_report)

    print("\n==============================")
    print("Drift Detection Completed")
    print("==============================")
    print(f"Statistical Report : {saved_path}")
    print(f"Evidently Report   : {evidently_report_path}")
    print("Decision            :", drift_report["summary"]["decision"])