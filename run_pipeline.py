
from pathlib import Path
import subprocess
import sys
import json
import shutil
from datetime import datetime, timezone
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STEPS = [
    (
        "Open-Meteo Data Ingestion",
        "src/data_ingestion/open_meteo_ingestion.py"
    ),
    (
        "API Data Validation",
        "src/data_ingestion/validate_api_data.py"
    ),
    (
        "PSI, KS and Evidently Analysis",
        "src/monitoring/api_drift_test.py"
    ),
    (
        "Decision Engine",
        "src/monitoring/api_decision.py"
    ),
    (
        "SHAP Feature Diagnosis",
        "src/diagnosis/api_feature_diagnosis.py"
    ),
    (
        "Model Health Report",
        "src/monitoring/api_health_report.py"
    ),
]

REQUIRED_REPORTS = [
    "artifacts/api_drift/api_drift_test.json",
    "artifacts/api_drift/api_evidently_results.json",
    "artifacts/api_drift/api_decision.json",
    "artifacts/api_drift/api_feature_diagnosis.json",
    "artifacts/api_drift/api_model_health_report.json",
]

EVIDENTLY_HTML = (
    "artifacts/api_drift/api_evidently_report.html"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_pipeline():

    started = datetime.now(timezone.utc)

    run_id = (
        started.strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:8]
    )

    print("\n================================")
    print("AUTOMATED MLOPS PIPELINE")
    print("================================")
    print("Pipeline Run ID:", run_id)

    # STEP 1: Clear old working reports
    # Previously archived runs are preserved.
    working_outputs = (
        REQUIRED_REPORTS + [EVIDENTLY_HTML]
    )

    for relative_path in working_outputs:

        path = PROJECT_ROOT / relative_path

        if path.exists():
            path.unlink()

    print("\nPrevious working reports cleared.")

    # STEP 2: Execute all six stages
    for name, script in STEPS:

        print(f"\nRunning: {name}", flush=True)

        script_path = PROJECT_ROOT / script

        if not script_path.is_file():
            raise FileNotFoundError(
                f"Pipeline script missing: {script_path}"
            )

        subprocess.run(
            [PYTHON, str(script_path)],
            cwd=PROJECT_ROOT,
            check=True
        )

        print(f"Completed: {name}", flush=True)

    # STEP 3: Verify fresh reports
    print("\nVerifying generated reports...")

    for relative_path in REQUIRED_REPORTS:

        report_path = PROJECT_ROOT / relative_path

        if not report_path.is_file():
            raise FileNotFoundError(
                f"Required report missing: {report_path}"
            )

        if report_path.stat().st_mtime < started.timestamp() - 2:
            raise RuntimeError(
                f"Report appears outdated: {report_path}"
            )

        load_json(report_path)

    html_path = PROJECT_ROOT / EVIDENTLY_HTML

    if not html_path.is_file():
        raise FileNotFoundError(
            f"Evidently HTML missing: {html_path}"
        )

    if html_path.stat().st_mtime < started.timestamp() - 2:
        raise RuntimeError(
            "Evidently HTML appears outdated."
        )

    print("All five JSON reports and HTML verified.")

    # STEP 4: Archive all reports
    run_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "pipeline_runs"
        / run_id
    )

    run_dir.mkdir(parents=True, exist_ok=False)

    for relative_path in working_outputs:

        source = PROJECT_ROOT / relative_path

        shutil.copy2(
            source,
            run_dir / source.name
        )

    print("Reports archived at:", run_dir)

    # STEP 5: Load health report
    health = load_json(
        run_dir / "api_model_health_report.json"
    )

    final_decision = health["decision"]["decision"]

    # STEP 6: Save metadata BEFORE payload generation
    metadata_path = run_dir / "run_metadata.json"

    metadata = {
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": None,
        "execution_time_sec": None,
        "status": "PROCESSING",
        "decision": final_decision,
        "reports": [
            path.name
            for path in run_dir.iterdir()
            if path.is_file()
        ]
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    # STEP 7: Generate payload using the CURRENT run ID
    payload_script = (
        PROJECT_ROOT
        / "src"
        / "monitoring"
        / "backend_payload.py"
    )

    print("\nGenerating backend payload...", flush=True)

    subprocess.run(
        [PYTHON, str(payload_script), run_id],
        cwd=PROJECT_ROOT,
        check=True
    )

    # STEP 8: Verify backend payload
    payload_path = run_dir / "backend_payload.json"

    if not payload_path.is_file():
        raise FileNotFoundError(
            f"Backend payload missing: {payload_path}"
        )

    payload = load_json(payload_path)

    if payload["run_id"] != run_id:
        raise ValueError(
            "Backend payload Run ID mismatch."
        )

    if payload["decision"]["decision"] != final_decision:
        raise ValueError(
            "Backend payload decision mismatch."
        )

    print("Backend payload verified:", payload_path)

    # STEP 9: Finalize run metadata
    finished = datetime.now(timezone.utc)

    metadata["finished_at"] = finished.isoformat()
    metadata["execution_time_sec"] = round(
        (finished - started).total_seconds(),
        2
    )
    metadata["status"] = "COMPLETED"

    metadata["reports"] = sorted(
        path.name
        for path in run_dir.iterdir()
        if path.is_file()
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    # STEP 10: Final summary
    print("\n================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("================================")

    print("Run ID:", run_id)
    print(
        "Execution Time:",
        metadata["execution_time_sec"],
        "seconds"
    )
    print("Final Decision:", final_decision)
    print("Reports Archived At:", run_dir)
    print("Backend Payload:", payload_path)
    print(
        "Health Report:",
        run_dir / "api_model_health_report.json"
    )


if __name__ == "__main__":
    run_pipeline()
