
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

API_DIR = PROJECT_ROOT / "artifacts" / "api_drift"

API_DRIFT_PATH = API_DIR / "api_drift_test.json"
API_DECISION_PATH = API_DIR / "api_decision.json"

SHAP_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "shap"
    / "diagnosis_report.json"
)

OUTPUT_PATH = API_DIR / "api_feature_diagnosis.json"


def generate_api_feature_diagnosis():

    with open(API_DRIFT_PATH, encoding="utf-8") as f:
        drift = json.load(f)

    with open(API_DECISION_PATH, encoding="utf-8") as f:
        decision = json.load(f)

    with open(SHAP_PATH, encoding="utf-8") as f:
        shap_report = json.load(f)

    importance = {
        item["feature"]: item["mean_abs_shap"]
        for item in shap_report["top_features"]
    }

    confirmed = {
        item["feature"]
        for item in decision["decision"][
            "confirmed_drift_features"
        ]
    }

    conflicts = set(
        decision["decision"]["conflicting_features"]
    )

    # Rank the available SHAP scores.
    # This is a heuristic, not a calibrated risk score.
    available_scores = sorted(
        importance.values(),
        reverse=True
    )

    median_importance = (
        available_scores[len(available_scores) // 2]
        if available_scores else 0
    )

    results = []

    for feature, metrics in drift["features"].items():

        severity = metrics["severity"]
        shap_value = importance.get(feature)

        if feature in conflicts:
            priority = "Manual Review"

        elif feature in confirmed:

            if shap_value is None:
                priority = "Review"

            elif shap_value >= median_importance:
                priority = "High"

            else:
                priority = "Investigate"

        else:
            priority = "No Drift"

        results.append({
            "feature": feature,
            "psi": metrics["psi"],
            "ks_pvalue": metrics["ks_pvalue"],
            "drift_status": severity,
            "evidently_drift": decision[
                "feature_evidence"
            ][feature]["evidently_drift"],
            "confirmed_drift": feature in confirmed,
            "mean_abs_shap": shap_value,
            "priority": priority
        })

    priority_order = {
        "High": 4,
        "Investigate": 3,
        "Manual Review": 2,
        "Review": 1,
        "No Drift": 0
    }

    results.sort(
        key=lambda item: (
            priority_order[item["priority"]],
            item["mean_abs_shap"]
            if item["mean_abs_shap"] is not None
            else -1
        ),
        reverse=True
    )

    report = {
        "data_source": "Open-Meteo",
        "decision": decision["decision"],
        "recommended_action": decision[
            "decision"
        ]["action"],
        "shap_source": "Historical sample",
        "high_priority_features": [
            item["feature"]
            for item in results
            if item["priority"] == "High"
        ],
        "conflicting_features": sorted(conflicts),
        "features": results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\nAPI Feature Diagnosis Completed")
    print("Decision:", report["decision"]["decision"])
    print("Report:", OUTPUT_PATH)

    for item in results:
        print(
            item["feature"],
            "| Drift:", item["drift_status"],
            "| SHAP:", item["mean_abs_shap"],
            "| Priority:", item["priority"]
        )


if __name__ == "__main__":
    generate_api_feature_diagnosis()
