
from pathlib import Path
from datetime import datetime, timezone
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

API_DIR = PROJECT_ROOT / "artifacts" / "api_drift"

OUTPUT_PATH = API_DIR / "api_model_health_report.json"


def load_json(filename):
    path = API_DIR / filename

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_api_health_report():

    drift = load_json("api_drift_test.json")
    decision = load_json("api_decision.json")
    diagnosis = load_json("api_feature_diagnosis.json")

    features = diagnosis["features"]

    high_priority = [
        item["feature"]
        for item in features
        if item["priority"] == "High"
    ]

    report = {
        "report_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "data_source": "Open-Meteo",

        "model": {
            "name": "XGBoost",
            "purpose": "RainTomorrow prediction"
        },

        "monitoring": {
            "reference_records": drift["reference_records"],
            "production_records": drift["api_records"],
            "stations": drift["stations"],
            "psi_status": decision["psi_status"],
            "minimum_ks_pvalue": decision[
                "minimum_ks_pvalue"
            ],
            "evidently_dataset_drift": decision[
                "evidently_dataset_drift"
            ],
            "evidently_drift_percentage": decision[
                "evidently_drift_percentage"
            ]
        },

        "decision": decision["decision"],

        "diagnosis": {
    "high_priority_features": high_priority,
    "conflicting_features": diagnosis.get(
        "conflicting_features", []
    ),
    "feature_details": features
},

        "limitations": [
            "This report measures data drift, not actual model accuracy.",
            "SHAP importance comes from a historical sample.",
            "The API comparison uses only three temperature features.",
            "The reference and API data come from different years and sources."
        ]
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\nAPI Model Health Report Generated")
    print("Report:", OUTPUT_PATH)
    print("Decision:", decision["decision"]["decision"])
    print("High Priority Features:", high_priority)


if __name__ == "__main__":
    generate_api_health_report()
