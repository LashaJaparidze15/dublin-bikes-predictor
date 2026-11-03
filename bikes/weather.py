"""Met Éireann weather: hourly history for training, point forecast for live predictions.

Both sources are turned into the same hourly table so the model sees identical inputs:
    time (UTC hour) | rain_mm | temp_c | wind_ms | humidity
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

from bikes.config import FORECAST_URL, WEATHER_DIR, WEATHER_HOURLY_URL

KNOTS_TO_MS = 0.514444
WEATHER_COLUMNS = ["rain_mm", "temp_c", "wind_ms", "humidity"]


def download_hourly(path: Path = WEATHER_DIR / "hly532.csv") -> Path:
    """Download the full Dublin Airport hourly record (about 60 MB, goes back to 1941)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(WEATHER_HOURLY_URL, timeout=300)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def load_hourly(path: Path = WEATHER_DIR / "hly532.csv", since: str = "2024-01-01") -> pd.DataFrame:
    """Read the Met Éireann hourly CSV. The file starts with a free-text key, so find the header row."""
    with open(path, encoding="latin-1") as fh:
        header_row = next(i for i, line in enumerate(fh) if line.startswith("date,"))
    df = pd.read_csv(path, skiprows=header_row, encoding="latin-1", low_memory=False)
    df["time"] = pd.to_datetime(df["date"], format="%d-%b-%Y %H:%M", utc=True)
    df = df[df["time"] >= pd.Timestamp(since, tz="UTC")]
    out = pd.DataFrame({
        "time": df["time"],
        "rain_mm": pd.to_numeric(df["rain"], errors="coerce"),
        "temp_c": pd.to_numeric(df["temp"], errors="coerce"),
        "wind_ms": pd.to_numeric(df["wdsp"], errors="coerce") * KNOTS_TO_MS,
        "humidity": pd.to_numeric(df["rhum"], errors="coerce"),
    })
    # Short gaps (a missed reading) are filled from neighbours, longer ones stay missing
    out = out.set_index("time").sort_index().interpolate(limit=3).reset_index()
    return out


def parse_forecast(xml_text: str) -> pd.DataFrame:
    """Turn Met Éireann's forecast XML into the hourly weather table.

    The feed alternates two kinds of <time> element: instant ones (from == to) carry temperature,
    wind and humidity; one-hour intervals (from < to) carry precipitation for that hour.
    """
    root = ET.fromstring(xml_text)
    instant, rain = {}, {}
    for t in root.iter("time"):
        start, end, loc = t.get("from"), t.get("to"), t.find("location")
        if start == end:
            temp, wind, hum = loc.find("temperature"), loc.find("windSpeed"), loc.find("humidity")
            instant[end] = {
                "temp_c": float(temp.get("value")) if temp is not None else None,
                "wind_ms": float(wind.get("mps")) if wind is not None else None,
                "humidity": float(hum.get("value")) if hum is not None else None,
            }
        else:
            precip = loc.find("precipitation")
            hours = (pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta(hours=1)
            if precip is not None and hours == 1:
                rain[end] = float(precip.get("value"))
    rows = [{"time": pd.Timestamp(ts), "rain_mm": rain.get(ts), **vals} for ts, vals in instant.items()]
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df[["time", *WEATHER_COLUMNS]]


def fetch_forecast() -> pd.DataFrame:
    """Get the current hourly forecast for Dublin city centre."""
    resp = requests.get(FORECAST_URL, timeout=60)
    resp.raise_for_status()
    return parse_forecast(resp.text)
