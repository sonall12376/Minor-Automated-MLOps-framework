from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGING_DIR = PROJECT_ROOT / "data" / "api_staging"

files = sorted(STAGING_DIR.glob("open_meteo_multi_*.csv"))

if not files:
    raise FileNotFoundError("No multi-station API batch found.")

df = pd.read_csv(files[-1])

print("\n========== API DATA VALIDATION ==========")

print("\nTotal Records:", len(df))
print("Total Stations:", df["station_name"].nunique())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Station-Date Records:")
print(df.duplicated(
    subset=["station_name", "date_of_record"]
).sum())

print("\nRecords Per Station:")
print(df.groupby("station_name").size())

print("\nNumerical Statistics:")
print(df.describe().round(2))

print("\nBasic Quality Checks:")

checks = {
    "No missing values": not df.isnull().any().any(),

    "No duplicate station-date records":
        not df.duplicated(
            subset=["station_name", "date_of_record"]
        ).any(),

    "Temperature ordering valid":
        (
            (df["min_temp"] <= df["avg_temp"])
            & (df["avg_temp"] <= df["max_temp"])
        ).all(),

    "Rainfall non-negative":
        (df["precipitation_sum_mm"] >= 0).all(),

    "Wind speed non-negative":
        (df["wind_speed_kmh"] >= 0).all(),

    "Pressure within broad plausible range":
        df["air_pressure_hpa"].between(850, 1100).all(),

    "Latitude valid":
        df["latitude"].between(-90, 90).all(),

    "Longitude valid":
        df["longitude"].between(-180, 180).all()
}

for name, passed in checks.items():
    print(
        f"{name}: {'PASS' if passed else 'FAIL'}"
    )

print("\n========================================")

if not all(checks.values()):
    raise ValueError(
        "API data failed validation. Do not promote to production."
    )

print("API data passed basic quality validation.")