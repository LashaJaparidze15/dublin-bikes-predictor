"""Read the live Dublin Bikes feed (GBFS v2.3)."""
import pandas as pd
import requests

from bikes.config import GBFS_INFO_URL, GBFS_STATUS_URL


def fetch_status() -> pd.DataFrame:
    """Current reading for every station, in the snapshots table layout.

    The feed's own `last_reported` is when the station last CHANGED, so a quiet station can
    show a time from hours ago even though its count is current. The history files have a
    reading every 5 to 10 minutes, so to match them each reading is stamped with the time it
    was fetched (the feed's `last_updated`) instead.
    """
    feed = requests.get(GBFS_STATUS_URL, timeout=30).json()
    df = pd.DataFrame(feed["data"]["stations"])
    fetched = pd.Timestamp(feed.get("last_updated") or pd.Timestamp.now(tz="UTC").timestamp(),
                           unit="s", tz="UTC")
    return pd.DataFrame({
        "station_id": df["station_id"].astype(int),
        "last_reported": fetched,
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
