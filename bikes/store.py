"""Where live data goes: Supabase Postgres when DATABASE_URL is set, local parquet otherwise.

The local fallback means the collector, predictor and dashboard all run on a laptop with no
database at all, which is handy for development and for anyone marking the portfolio.
"""
import os
from datetime import datetime, timezone

import pandas as pd

from bikes.config import DATA_DIR

LIVE_DIR = DATA_DIR / "live"


def _db_url() -> str | None:
    return os.environ.get("DATABASE_URL")


def _connect():
    import psycopg  # imported here so the local mode does not need the driver
    return psycopg.connect(_db_url(), autocommit=True)


def _query(sql: str, params: dict) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame (plain psycopg, so pandas does not warn about it)."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def save_stations(stations: pd.DataFrame) -> None:
    if not _db_url():
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        stations.to_parquet(LIVE_DIR / "stations.parquet", index=False)
        return
    rows = stations[["station_id", "name", "address", "lat", "lon", "capacity"]].itertuples(index=False)
    with _connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """insert into stations (station_id, name, address, lat, lon, capacity)
               values (%s, %s, %s, %s, %s, %s)
               on conflict (station_id) do update set name = excluded.name, address = excluded.address,
                   lat = excluded.lat, lon = excluded.lon, capacity = excluded.capacity, updated_at = now()""",
            [tuple(r) for r in rows])


def save_snapshots(snaps: pd.DataFrame) -> int:
    """Store one collection run. Returns the number of rows written."""
    if not _db_url():
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = LIVE_DIR / f"snapshots_{day}.parquet"
        if path.exists():
            snaps = pd.concat([pd.read_parquet(path), snaps])
        snaps.drop_duplicates(["station_id", "last_reported"]).to_parquet(path, index=False)
        return len(snaps)
    cols = ["station_id", "last_reported", "bikes", "docks", "is_installed", "is_renting",
            "is_returning", "capacity"]
    with _connect() as conn, conn.cursor() as cur:
        cur.executemany(
            f"""insert into snapshots ({", ".join(cols)}) values ({", ".join(["%s"] * len(cols))})
                on conflict do nothing""",
            [tuple(r) for r in snaps[cols].itertuples(index=False)])
    return len(snaps)


def recent_snapshots(minutes: int = 80) -> pd.DataFrame:
    """Readings from the last `minutes`, in the same column layout as the history files."""
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes)
    if not _db_url():
        files = sorted(LIVE_DIR.glob("snapshots_*.parquet"))[-2:]
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files])
        df = df[df["last_reported"] >= since]
    else:
        df = _query("select * from snapshots where last_reported >= %(since)s",
                    {"since": since.to_pydatetime()})
    return df.rename(columns={"bikes": "num_bikes_available", "docks": "num_docks_available"})


def save_predictions(preds: pd.DataFrame) -> None:
    if not _db_url():
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        path = LIVE_DIR / "predictions.parquet"
        if path.exists():
            preds = pd.concat([pd.read_parquet(path), preds])
        preds.to_parquet(path, index=False)
        return
    cols = ["made_at", "station_id", "horizon_min", "target_time", "bikes_now",
            "predicted_bikes", "p_empty", "p_full", "model_version"]
    with _connect() as conn, conn.cursor() as cur:
        cur.executemany(
            f"""insert into predictions ({", ".join(cols)}) values ({", ".join(["%s"] * len(cols))})
                on conflict do nothing""",
            [tuple(r) for r in preds[cols].astype(object).itertuples(index=False)])


def prediction_accuracy(days: int = 14) -> pd.DataFrame:
    """Predictions joined to what really happened (only works with the database)."""
    if not _db_url():
        return pd.DataFrame()
    return _query("select * from prediction_accuracy where made_at >= now() - make_interval(days => %(d)s)",
                  {"d": days})
