
from pathlib import Path
import json

import pandas as pd

from drift_metrics import (
    calculate_psi,
    calculate_ks_test,
    get_drift_severity
)

from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "weather_preprocessed.csv"
)

API_DIR = PROJECT_ROOT / "data" / "api_staging"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "api_drift"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["avg_temp", "min_temp", "max_temp"]


def run_api_drift_test():

    historical = pd.read_csv(
        HISTORICAL_PATH,
        parse_dates=["date_of_record"]
    )

    files = sorted(
        API_DIR.glob("open_meteo_multi_*.csv")
    )

    if not files:
        raise FileNotFoundError(
            "No Open-Meteo batch found."
        )

    api = pd.read_csv(files[-1])

    api["date_of_record"] = pd.to_datetime(
        api["date_of_record"]
    )

    # Use the same stations and month as API data
    # Match historical observations to the API's calendar dates
    api_dates = set(
        api["date_of_record"].dt.strftime("%m-%d")
)
    reference = historical[
        historical["station_name"].isin(
            api["station_name"].unique()
    )
    & historical["date_of_record"].dt.strftime(
        "%m-%d"
        ).isin(api_dates)
        ].copy()
    
    if reference.empty:
        raise ValueError("No matching historical data found.")
    # Balance historical records across stations
    station_counts = reference.groupby("station_name").size()
    api_stations = set(api["station_name"].unique())
    if set(station_counts.index) != api_stations:
        raise ValueError("Some API stations are missing.")
    samples_per_station = int(station_counts.min())
    
    reference = (
        reference.groupby("station_name", group_keys=False)
        .sample(n=samples_per_station, random_state=42)
        .reset_index(drop=True)
        )
    
    print("Balanced historical records:", len(reference))
    print("Records per station:", samples_per_station)

# Use an equal number of historical records per station
    station_counts = reference.groupby("station_name").size()
    
    api_stations = set(api["station_name"].unique())
    
    if set(station_counts.index) != api_stations:
        raise ValueError(
            "Some API stations have no historical reference records."
            )
        
        samples_per_station = int(station_counts.min())
        reference = (
            reference.groupby("station_name", group_keys=False)
            .sample(n=samples_per_station, random_state=42)
            .reset_index(drop=True)
            )
        
        print("Balanced historical records:", len(reference))
        print("Records per station:", samples_per_station)

    results = {}

    for feature in FEATURES:

        ref = reference[feature].dropna()
        current = api[feature].dropna()

        psi = calculate_psi(ref, current)

        ks_stat, ks_pvalue = calculate_ks_test(
            ref, current
        )

        results[feature] = {
            "psi": round(float(psi), 4),
            "ks_statistic": round(
                float(ks_stat), 4
            ),
            "ks_pvalue": round(
                float(ks_pvalue), 6
            ),
            "severity": get_drift_severity(psi)
        }

        # Independent Evidently analysis
    reference_features = reference[FEATURES].copy()
    current_features = api[FEATURES].copy()

    evidently_report = Report(
        metrics=[DataDriftPreset()]
    )

    snapshot = evidently_report.run(
        reference_data=reference_features,
        current_data=current_features
    )

    evidently_html = OUTPUT_DIR / "api_evidently_report.html"
    evidently_json = OUTPUT_DIR / "api_evidently_results.json"

    snapshot.save_html(str(evidently_html))

    with open(evidently_json, "w", encoding="utf-8") as f:
        json.dump(snapshot.dict(), f, indent=4, default=str)

    print("\nEvidently HTML:", evidently_html)
    print("Evidently JSON:", evidently_json)

    report = {
        "comparison": "Historical October vs Open-Meteo",
        "reference_records": len(reference),
        "api_records": len(api),
        "stations": api["station_name"].nunique(),
        "features": results,
        "note": (
            "Exploratory drift comparison. "
            "Not a production drift alert."
        )
    }

    output_path = OUTPUT_DIR / "api_drift_test.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\nAPI Drift Test Completed")
    print("Reference records:", len(reference))
    print("API records:", len(api))

    for feature, metrics in results.items():
        print(
            f"{feature}: "
            f"PSI={metrics['psi']}, "
            f"KS p={metrics['ks_pvalue']}, "
            f"Status={metrics['severity']}"
        )

    print("\nReport:", output_path)


if __name__ == "__main__":
    run_api_drift_test()
