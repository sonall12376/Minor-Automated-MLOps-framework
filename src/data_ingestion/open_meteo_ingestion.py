from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "api_staging"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://api.open-meteo.com/v1/forecast"

STATIONS = {
    "Agartala": (23.8833, 91.2500),
    "New Delhi / Safdarjung": (28.5833, 77.2000),
    "Bombay / Santacruz": (19.1167, 72.8500),
    "Calcutta / Alipore": (22.5333, 88.3333),
    "Madras": (13.0667, 80.2500),
    "Bangalore": (12.9667, 77.5833),
    "Jaipur / Sanganer": (26.8167, 75.8000),
    "Lucknow": (26.8667, 80.9333),
    "Hyderabad Airport": (17.4500, 78.4667),
    "Poona": (18.5333, 73.8500)
}


def fetch_station_weather(station_name, latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "pressure_msl",
            "wind_speed_10m"
        ],
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "precipitation_sum"
        ],
        "timezone": "Asia/Kolkata",
        "past_days": 7,
        "forecast_days": 0,
        "wind_speed_unit": "kmh"
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()

    hourly = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "air_pressure_hpa": data["hourly"]["pressure_msl"],
        "wind_speed_kmh": data["hourly"]["wind_speed_10m"]
    })

    hourly["date_of_record"] = (
        hourly["time"].dt.strftime("%Y-%m-%d")
    )

    daily_means = (
        hourly.groupby("date_of_record")[
            ["air_pressure_hpa", "wind_speed_kmh"]
        ]
        .mean()
        .reset_index()
    )

    daily = data["daily"]

    df = pd.DataFrame({
        "date_of_record": daily["time"],
        "station_name": station_name,
        "avg_temp": daily["temperature_2m_mean"],
        "min_temp": daily["temperature_2m_min"],
        "max_temp": daily["temperature_2m_max"],
        "precipitation_sum_mm": daily["precipitation_sum"],
        "latitude": data["latitude"],
        "longitude": data["longitude"],
        "elevation_m": data.get("elevation")
    })

    return df.merge(
        daily_means,
        on="date_of_record",
        how="left",
        validate="one_to_one"
    )


def fetch_all_stations():

    frames = []

    for station, (latitude, longitude) in STATIONS.items():

        print(f"Fetching weather for {station}...")

        try:
            df = fetch_station_weather(
                station,
                latitude,
                longitude
            )
            frames.append(df)

        except (requests.RequestException, ValueError, KeyError) as error:
            print(f"Failed for {station}: {error}")

    if not frames:
        raise RuntimeError("No station data was fetched.")

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_path = (
        OUTPUT_DIR / f"open_meteo_multi_{timestamp}.csv"
    )

    combined.to_csv(output_path, index=False)

    print("\nMulti-Station API Fetch Completed")
    print("Stations:", combined["station_name"].nunique())
    print("Total rows:", len(combined))
    print("Saved:", output_path)
    print("\nSample Data:")
    print(combined.head().to_string(index=False))

    return combined


if __name__ == "__main__":
    fetch_all_stations()