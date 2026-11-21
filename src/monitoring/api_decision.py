
from pathlib import Path
import json
import re

from decision_engine import make_feature_based_decision

PROJECT_ROOT = Path(__file__).resolve().parents[2]

API_DIR = PROJECT_ROOT / "artifacts" / "api_drift"

DRIFT_PATH = API_DIR / "api_drift_test.json"
EVIDENTLY_PATH = API_DIR / "api_evidently_results.json"
OUTPUT_PATH = API_DIR / "api_decision.json"


def run_api_decision():

    with open(DRIFT_PATH, encoding="utf-8") as f:
        drift = json.load(f)

    with open(EVIDENTLY_PATH, encoding="utf-8") as f:
        evidently = json.load(f)

    feature_results = {}
    evidently_features = {}

    # Extract actual feature-level Evidently results
    for metric in evidently.get("metrics", []):

        metric_name = metric.get("metric_name", "")

        if metric_name.startswith("ValueDrift("):

            match = re.search(
                r"column=([^,)]+)",
                metric_name
            )

            if not match:
                continue

            feature = match.group(1).strip()

            pvalue = float(metric["value"])

            threshold_match = re.search(
                r"threshold=([0-9.eE+-]+)",
                metric_name
            )

            if not threshold_match:
                raise ValueError(
                    f"Evidently threshold missing: {feature}"
                )

            threshold = float(
                threshold_match.group(1)
            )

            evidently_features[feature] = (
                pvalue < threshold
            )

    # Check that all monitored features have Evidently results
    for feature, metrics in drift["features"].items():

        if feature not in evidently_features:
            raise ValueError(
                f"Evidently result missing for {feature}"
            )

        feature_results[feature] = {
            "severity": metrics["severity"],
            "ks_pvalue": metrics["ks_pvalue"],
            "evidently_drift": evidently_features[feature]
        }

    # Make feature-level decision
    decision = make_feature_based_decision(
        feature_results
    )

    # Preserve the existing report structure
    # for downstream diagnosis and health reports
    report = {
        "data_source": "Open-Meteo",
        "comparison": drift["comparison"],
        "psi_status": max(
            (
                item["severity"]
                for item in drift["features"].values()
            ),
            key=lambda status: {
                "No Drift": 0,
                "Low Drift": 1,
                "Moderate Drift": 2,
                "High Drift": 3
            }[status]
        ),
        "minimum_ks_pvalue": min(
            item["ks_pvalue"]
            for item in drift["features"].values()
        ),
        "evidently_dataset_drift": None,
        "evidently_drifted_features": sum(
            evidently_features.values()
        ),
        "evidently_drift_percentage": round(
            100 * sum(evidently_features.values())
            / len(evidently_features),
            2
        ),
        "decision": decision,
        "feature_evidence": feature_results
    }

    # Read actual Evidently dataset-level result
    for metric in evidently.get("metrics", []):

        if metric.get("metric_name", "").startswith(
            "DriftedColumnsCount("
        ):
            share = float(metric["value"]["share"])

            threshold_match = re.search(
                r"drift_share=([0-9.eE+-]+)",
                metric["metric_name"]
            )

            if not threshold_match:
                raise ValueError(
                    "Evidently dataset threshold missing"
                )

            threshold = float(
                threshold_match.group(1)
            )

            report["evidently_dataset_drift"] = (
                share >= threshold
            )

            break

    if report["evidently_dataset_drift"] is None:
        raise ValueError(
            "Evidently dataset drift summary missing"
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\nAPI Feature-Based Decision Completed")
    print("Final Decision:", decision["decision"])
    print("Confirmed Features:",
          decision["confirmed_drift_features"])
    print("Conflicting Features:",
          decision["conflicting_features"])
    print("Report:", OUTPUT_PATH)


if __name__ == "__main__":
    run_api_decision()
