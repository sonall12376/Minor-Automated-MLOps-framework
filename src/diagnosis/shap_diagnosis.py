from pathlib import Path
import sys
import json

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src" / "training")
)

from train_model import preprocess_data


MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "baseline_model.pkl"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_preprocessed.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "shap"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


def generate_shap_report(sample_size=200):

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Load sample of preprocessed data
    df = pd.read_csv(DATA_PATH)

    df = df.sample(
        n=min(sample_size, len(df)),
        random_state=42
    )

    # Apply same feature engineering as training
    df = preprocess_data(df)

    # Align features with saved model
    feature_names = list(model.feature_names_in_)

    X = df.drop(
        columns=["RainTomorrow"],
        errors="ignore"
    )

    X = X.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Explain model predictions
    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X)

    # Save SHAP bar plot
    plt.figure()

    shap.plots.bar(
        shap_values,
        max_display=15,
        show=False
    )

    plt.savefig(
        OUTPUT_PATH / "shap_bar.png",
        bbox_inches="tight",
        dpi=150
    )

    plt.close("all")

    # Save SHAP beeswarm plot
    shap.plots.beeswarm(
        shap_values,
        max_display=15,
        show=False
    )

    plt.savefig(
        OUTPUT_PATH / "shap_summary.png",
        bbox_inches="tight",
        dpi=150
    )

    plt.close("all")

    # Calculate mean absolute SHAP importance
    importance = np.abs(
        shap_values.values
    ).mean(axis=0)

    ranked_features = sorted(
        zip(feature_names, importance),
        key=lambda item: item[1],
        reverse=True
    )

    report = {
        "model": "XGBoost",
        "sample_size": len(X),
        "total_features": len(feature_names),
        "top_features": [
            {
                "feature": name,
                "mean_abs_shap": round(
                    float(value), 6
                )
            }
            for name, value in ranked_features[:15]
        ]
    }

    report_path = OUTPUT_PATH / "diagnosis_report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\nSHAP Diagnosis Completed")
    print("Report:", report_path)
    print("Plots saved inside:", OUTPUT_PATH)


if __name__ == "__main__":
    generate_shap_report()