"""Read the live Dublin Bikes feed (GBFS v2.3)."""
import pandas as pd
import requests

from bikes.config import GBFS_INFO_URL, GBFS_STATUS_URL


def fetch_status() -> pd.DataFrame:
    """Current reading for every station, in the snapshots table layout."""
    stations = requests.get(GBFS_STATUS_URL, timeout=30).json()["data"]["stations"]
    df = pd.DataFrame(stations)
    return pd.DataFrame({
        "station_id": df["station_id"].astype(int),
        "last_reported": pd.to_datetime(df["last_reported"], unit="s", utc=True),
        "bikes": df["num_bikes_available"].astype(int),
        "docks": df["num_docks_available"].astype(int),
        "is_installed": df["is_installed"].astype(bool),
        "is_renting": df["is_renting"].astype(bool),
        "is_returning": df["is_returning"].astype(bool),
        # GBFS status has no capacity field, so use bikes + docks + disabled docks
        "capacity": (df["num_bikes_available"] + df["num_docks_available"]
                     + df.get("num_docks_disabled", 0)).astype(int),
    })


def fetch_stations() -> pd.DataFrame:
    """Name, address, location and capacity of every station."""
    stations = requests.get(GBFS_INFO_URL, timeout=30).json()["data"]["stations"]
    df = pd.DataFrame(stations)
    df["station_id"] = df["station_id"].astype(int)
    return df[["station_id", "name", "address", "lat", "lon", "capacity"]]
