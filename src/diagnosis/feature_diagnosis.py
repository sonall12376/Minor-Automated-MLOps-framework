from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DRIFT_REPORT = (
    PROJECT_ROOT / "artifacts" / "drift_reports"
    / "drift_report.json"
)

SHAP_REPORT = (
    PROJECT_ROOT / "artifacts" / "shap"
    / "diagnosis_report.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "artifacts" / "shap"
    / "feature_diagnosis.json"
)


def generate_feature_diagnosis():

    with open(DRIFT_REPORT, "r", encoding="utf-8") as f:
        drift_data = json.load(f)

    with open(SHAP_REPORT, "r", encoding="utf-8") as f:
        shap_data = json.load(f)

    feature_drift = drift_data["feature_drift"]

    shap_importance = {
        item["feature"]: item["mean_abs_shap"]
        for item in shap_data["top_features"]
    }

    diagnosis = []

    for feature, metrics in feature_drift.items():

        is_drifted = metrics["drift_detected"]

        importance = shap_importance.get(feature)

        if is_drifted and importance is not None:
            priority = "High"

        elif is_drifted:
            priority = "Review"

        else:
            priority = "No Drift"

        diagnosis.append({
            "feature": feature,
            "psi": metrics["psi"],
            "ks_pvalue": metrics["ks_pvalue"],
            "drift_detected": is_drifted,
            "mean_abs_shap": importance,
            "priority": priority
        })

    diagnosis.sort(
        key=lambda x: (
            x["drift_detected"],
            x["mean_abs_shap"]
            if x["mean_abs_shap"] is not None else -1
        ),
        reverse=True
    )

    report = {
        "overall_decision": drift_data["summary"]["decision"],
        "total_features_analyzed": len(diagnosis),
        "high_priority_features": sum(
            item["priority"] == "High"
            for item in diagnosis
        ),
        "features": diagnosis
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\nFeature Diagnosis Completed")
    print("Report:", OUTPUT_PATH)
    print("High Priority Features:", report["high_priority_features"])

    return report


if __name__ == "__main__":
    generate_feature_diagnosis()