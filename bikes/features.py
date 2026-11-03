"""Turn raw station snapshots into a regular grid and then into model features.

Pipeline:
    raw snapshots (irregular, 5 to 10 min)  ->  build_grid()  ->  one row per station per 10 min
    grid + weather                          ->  make_features()  ->  features + targets per horizon

Timestamps are kept in UTC. Hour-of-day features use Irish local time, because people
commute by the clock on the wall, not by UTC.
"""
import numpy as np
import pandas as pd

from bikes.config import HISTORY_DIR, SLOT_MINUTES, TIMEZONE

HORIZONS = [10, 30, 60]          # minutes ahead we predict
LAGS = [10, 30, 60]              # minutes back we look

# Irish public holidays in the period covered by the data
PUBLIC_HOLIDAYS = pd.to_datetime([
    "2025-01-01", "2025-02-03", "2025-03-17", "2025-04-21", "2025-05-05", "2025-06-02",
    "2025-08-04", "2025-10-27", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-02-02", "2026-03-17", "2026-04-06", "2026-05-04", "2026-06-01",
    "2026-08-03", "2026-10-26", "2026-12-25", "2026-12-26",
]).date

NUMERIC_FEATURES = [
    "bikes", "docks", "capacity", "fill_ratio",
    *[f"bikes_lag{m}" for m in LAGS], "change_30", "change_60",
    "hour_sin", "hour_cos", "dow", "is_weekend", "is_holiday",
    "rain_mm", "rain_prev_mm", "temp_c", "wind_ms",
    "lat", "lon",
]
CATEGORICAL_FEATURES = ["station_id"]


def load_history(months: list[str] | None = None) -> pd.DataFrame:
    """Read the monthly parquet files written by bikes.history (all of them by default)."""
    files = sorted(HISTORY_DIR.glob("20??-??.parquet"))
    if months:
        files = [f for f in files if f.stem in months]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_stations() -> pd.DataFrame:
    """Latest known name and location for every station."""
    files = sorted(HISTORY_DIR.glob("stations_*.parquet"))
    stations = pd.concat([pd.read_parquet(f) for f in files])
    stations["station_id"] = stations["station_id"].astype(int)
    return stations.drop_duplicates("station_id", keep="last").reset_index(drop=True)


def build_grid(snapshots: pd.DataFrame, max_fill_slots: int = 2) -> pd.DataFrame:
    """Resample irregular snapshots onto a fixed 10-minute grid per station.

    - Rows from stations that are not installed or not renting are dropped (they report 0 bikes,
      which would teach the model that "broken" means "empty").
    - Within a slot the latest reading wins.
    - A missing slot is filled from the previous reading for up to `max_fill_slots` slots;
      longer gaps stay NaN so they never become fake training targets.
    """
    df = snapshots[snapshots["is_installed"] & snapshots["is_renting"]].copy()
    df["time"] = pd.to_datetime(df["last_reported"]).dt.floor(f"{SLOT_MINUTES}min")
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    df = (df.sort_values("last_reported")
            .groupby(["station_id", "time"], as_index=False)
            .last()[["station_id", "time", "num_bikes_available", "num_docks_available", "capacity"]]
            .rename(columns={"num_bikes_available": "bikes", "num_docks_available": "docks"}))
    if df.empty:  # e.g. overnight, when the whole scheme is closed
        return df

    full_range = pd.date_range(df["time"].min(), df["time"].max(), freq=f"{SLOT_MINUTES}min")
    pieces = []
    for station_id, grp in df.groupby("station_id"):
        grp = grp.set_index("time").reindex(full_range).ffill(limit=max_fill_slots)
        grp["station_id"] = station_id
        pieces.append(grp.rename_axis("time").reset_index())
    grid = pd.concat(pieces, ignore_index=True).dropna(subset=["bikes"])
    for col in ["bikes", "docks", "capacity"]:
        grid[col] = grid[col].astype("int16")
    return grid


def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Hour of day (as a circle, so 23:50 sits next to 00:00), weekday, weekend and holiday flags."""
    local = df[time_col].dt.tz_convert(TIMEZONE)
    hour = local.dt.hour + local.dt.minute / 60
    df["hour"] = hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow"] = local.dt.day_of_week
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["is_holiday"] = local.dt.date.isin(PUBLIC_HOLIDAYS).astype("int8")
    return df


def add_weather(df: pd.DataFrame, weather: pd.DataFrame, at_col: str = "target_time") -> pd.DataFrame:
    """Attach the weather for the hour we are predicting into, plus rain in the hour before now.

    In training this uses observed weather, which stands in for a perfect forecast. Live
    predictions use Met Éireann's forecast instead (see bikes.predict).
    """
    w = weather.set_index("time").sort_index()
    target_hour = df[at_col].dt.floor("h")
    df["rain_mm"] = w["rain_mm"].reindex(target_hour).to_numpy()
    df["temp_c"] = w["temp_c"].reindex(target_hour).to_numpy()
    df["wind_ms"] = w["wind_ms"].reindex(target_hour).to_numpy()
    df["rain_prev_mm"] = w["rain_mm"].reindex(df["time"].dt.floor("h") - pd.Timedelta(hours=1)).to_numpy()
    return df


def add_lags(grid: pd.DataFrame) -> pd.DataFrame:
    """Bikes 10/30/60 minutes ago at the same station, and how much that has changed since."""
    grid = grid.sort_values(["station_id", "time"]).reset_index(drop=True)
    by_station = grid.groupby("station_id")
    for minutes in LAGS:
        shifted = by_station[["time", "bikes"]].shift(minutes // SLOT_MINUTES)
        # only trust the lag if the slot really is `minutes` earlier (no gap in between)
        ok = (grid["time"] - shifted["time"]) == pd.Timedelta(minutes=minutes)
        grid[f"bikes_lag{minutes}"] = shifted["bikes"].where(ok)
    grid["change_30"] = grid["bikes"] - grid["bikes_lag30"]
    grid["change_60"] = grid["bikes"] - grid["bikes_lag60"]
    grid["fill_ratio"] = grid["bikes"] / (grid["bikes"] + grid["docks"]).clip(lower=1)
    return grid


def add_targets(grid: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """What actually happened `horizon` minutes later: bike count, and empty/full flags."""
    steps = horizon // SLOT_MINUTES
    future = grid.groupby("station_id")[["time", "bikes", "docks"]].shift(-steps)
    ok = (future["time"] - grid["time"]) == pd.Timedelta(minutes=horizon)
    grid["target_time"] = grid["time"] + pd.Timedelta(minutes=horizon)
    grid["target_bikes"] = future["bikes"].where(ok)
    grid["target_empty"] = (future["docks"].where(ok).notna() & (future["bikes"] == 0)).astype("int8")
    grid["target_full"] = (future["docks"].where(ok).notna() & (future["docks"] == 0)).astype("int8")
    # 0 = empty, 1 = some bikes and some docks, 2 = full
    grid["target_status"] = np.select([grid["target_empty"] == 1, grid["target_full"] == 1], [0, 2], 1)
    return grid


def make_features(grid: pd.DataFrame, weather: pd.DataFrame, stations: pd.DataFrame,
                  horizon: int = 30, with_targets: bool = True) -> pd.DataFrame:
    """Full feature table for one prediction horizon.

    station_id becomes a categorical over every known station (from `stations`), so the
    category codes are identical in training and in live prediction.
    """
    df = add_lags(grid)
    if with_targets:
        df = add_targets(df, horizon)
        df = df.dropna(subset=["target_bikes"])
    else:
        df["target_time"] = df["time"] + pd.Timedelta(minutes=horizon)
    df = add_time_features(df)
    df = add_weather(df, weather)
    df = df.merge(stations[["station_id", "lat", "lon"]], on="station_id", how="left")
    df["station_id"] = pd.Categorical(df["station_id"], categories=sorted(stations["station_id"]))
    return df
