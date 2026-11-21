from pathlib import Path
from datetime import datetime, timezone
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DRIFT_PATH = (
    PROJECT_ROOT / "artifacts" / "drift_reports"
    / "drift_report.json"
)

SHAP_PATH = (
    PROJECT_ROOT / "artifacts" / "shap"
    / "diagnosis_report.json"
)

DIAGNOSIS_PATH = (
    PROJECT_ROOT / "artifacts" / "shap"
    / "feature_diagnosis.json"
)

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "health_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required report missing: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_health_report():

    drift = load_json(DRIFT_PATH)
    shap = load_json(SHAP_PATH)
    diagnosis = load_json(DIAGNOSIS_PATH)

    summary = drift["summary"]

    report = {
        "report_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "model": {
            "name": shap["model"],
            "total_features": shap["total_features"],
            "shap_sample_size": shap["sample_size"]
        },

        "monitoring": {
            "overall_status": summary["overall_status"],
            "health_score": summary["health_score"],
            "total_features": summary["total_features"],
            "drifted_features": summary["drifted_features"],
            "average_psi": summary["average_psi"],
            "minimum_ks_pvalue": summary["minimum_ks_pvalue"],
            "evidently_dataset_drift": summary[
                "evidently_dataset_drift"
            ],
            "evidently_drift_percentage": summary[
                "evidently_drift_percentage"
            ]
        },

        "decision": {
            "status": summary["decision"],
            "confidence": summary["confidence"],
            "recommended_action": summary[
                "recommended_action"
            ]
        },

        "explainability": {
            "top_features": shap["top_features"][:10],
            "high_priority_features": diagnosis[
                "high_priority_features"
            ],
            "feature_diagnosis": diagnosis["features"]
        },

        "limitations": [
            "Health score is based on feature drift, not measured model accuracy.",
            "SHAP importance was calculated on historical sampled data.",
            "This report does not establish the cause of performance degradation."
        ]
    }

    output_path = OUTPUT_DIR / "model_health_report.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nModel Health Report Generated")
    print("Report:", output_path)
    print("Monitoring Status:", summary["overall_status"])
    print("Decision:", summary["decision"])

    return report


if __name__ == "__main__":
    generate_health_report()