
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "pipeline_runs"
)


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file missing: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_backend_payload(run_id=None):

    # Step 1: Select the correct pipeline run
    if run_id is not None:

        run_dir = RUNS_DIR / run_id

        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"Pipeline run not found: {run_dir}"
            )

    else:

        if not RUNS_DIR.is_dir():
            raise FileNotFoundError(
                "Pipeline runs directory not found."
            )

        run_dirs = sorted(
            path
            for path in RUNS_DIR.iterdir()
            if path.is_dir()
            and (path / "run_metadata.json").is_file()
        )

        if not run_dirs:
            raise FileNotFoundError(
                "No archived pipeline runs found."
            )

        run_dir = run_dirs[-1]

    # Step 2: Load metadata and reports
    metadata = load_json(
        run_dir / "run_metadata.json"
    )

    health = load_json(
        run_dir / "api_model_health_report.json"
    )

    drift = load_json(
        run_dir / "api_drift_test.json"
    )

    decision = load_json(
        run_dir / "api_decision.json"
    )

    diagnosis = load_json(
        run_dir / "api_feature_diagnosis.json"
    )

    # Step 3: Validate run consistency
    if metadata["run_id"] != run_dir.name:
        raise ValueError(
            "Run ID does not match archived folder."
        )

    final_decision = decision["decision"]["decision"]

    if health["decision"]["decision"] != final_decision:
        raise ValueError(
            "Health report and decision report disagree."
        )

    if diagnosis["decision"]["decision"] != final_decision:
        raise ValueError(
            "Diagnosis report and decision report disagree."
        )

    # Step 4: Construct backend-ready payload
    payload = {
        "schema_version": "1.0",

        "run_id": metadata["run_id"],

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "data_source": "Open-Meteo",

        "model": health["model"],

        "batch": {
            "reference_records": drift["reference_records"],
            "production_records": drift["api_records"],
            "stations": drift["stations"]
        },

        "monitoring": health["monitoring"],

        "decision": decision["decision"],

        "diagnosis": health["diagnosis"],

        "feature_evidence": decision["feature_evidence"],

        "feature_metrics": drift["features"],

        "report_paths": {
            "health_report": str(
                run_dir / "api_model_health_report.json"
            ),
            "evidently_html": str(
                run_dir / "api_evidently_report.html"
            )
        },

        "limitations": health.get("limitations", [])
    }

    # Step 5: Save payload in the same run folder
    output_path = run_dir / "backend_payload.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(payload, file, indent=4)

    # Step 6: Display results
    print("\n================================")
    print("BACKEND PAYLOAD GENERATED")
    print("================================")

    print("Run ID:", payload["run_id"])
    print("Decision:", final_decision)
    print("Output:", output_path)

    return output_path


if __name__ == "__main__":

    run_id = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    generate_backend_payload(run_id)
