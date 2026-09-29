from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "results"
)

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


def select_best_model(results):

    best_model = max(
        results,
        key=lambda model: results[model]["f1_score"]
    )

    return best_model


def save_model_comparison(results):

    output_path = (
        RESULTS_PATH
        / "model_comparison.json"
    )

    serializable_results = {}

    for model_name, metrics in results.items():

        serializable_results[model_name] = {}

        for key, value in metrics.items():

            serializable_results[model_name][key] = float(value)

    with open(output_path, "w") as file:

        json.dump(
            serializable_results,
            file,
            indent=4
        )

    return output_path