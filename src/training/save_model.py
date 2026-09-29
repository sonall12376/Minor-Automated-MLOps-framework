from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

MODEL_PATH.mkdir(
    parents=True,
    exist_ok=True
)


def save_model(
    model,
    model_name
):

    output_path = (
        MODEL_PATH
        / f"{model_name}.pkl"
    )

    joblib.dump(
        model,
        output_path
    )

    return output_path