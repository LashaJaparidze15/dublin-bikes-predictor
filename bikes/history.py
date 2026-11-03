"""Download the monthly Dublin Bikes history CSVs and store them as compact parquet files.

Each monthly CSV is 100+ MB because station name, address and coordinates repeat on every
row. We keep only the columns that change over time and save station details separately.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import requests

from bikes.config import CKAN_PACKAGE_URL, HISTORY_DIR

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}

STATUS_COLUMNS = ["last_reported", "station_id", "num_bikes_available", "num_docks_available",
                  "is_installed", "is_renting", "is_returning", "capacity"]
STATION_COLUMNS = ["station_id", "name", "address", "lat", "lon", "capacity"]


def list_monthly_files() -> dict[str, str]:
    """Return {"YYYY-MM": csv_url} for every monthly file in the GBFS-style format (May 2024 on)."""
    resources = requests.get(CKAN_PACKAGE_URL, timeout=60).json()["result"]["resources"]
    files = {}
    for res in resources:
        match = re.search(r"Station Data (\w+) (\d{4})", res["name"])
        if match and match.group(1).lower() in MONTHS:
            month = f"{match.group(2)}-{MONTHS[match.group(1).lower()]:02d}"
            files[month] = res["url"]
    return dict(sorted(files.items()))


def download_month(month: str, url: str, out_dir: Path = HISTORY_DIR) -> Path | None:
    """Download one month, keep the useful columns, and write data/history/<month>.parquet."""
    out_path = out_dir / f"{month}.parquet"
    if out_path.exists():
        return out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_dir / f"{month}.csv.part"
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp_csv, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    df = pd.read_csv(tmp_csv, dtype={"station_id": str})
    tmp_csv.unlink()
    if df.empty:
        print(f"{month}: file is empty, skipped")
        return None

    status = df[STATUS_COLUMNS].copy()
    status["last_reported"] = pd.to_datetime(status["last_reported"])
    status["station_id"] = status["station_id"].astype(int)
    status.to_parquet(out_path, index=False)

    stations = df[STATION_COLUMNS].drop_duplicates("station_id", keep="last")
    stations.to_parquet(out_dir / f"stations_{month}.parquet", index=False)
    print(f"{month}: {len(status):,} rows")
    return out_path


def main(start: str = "2025-01", end: str = "2099-12") -> None:
    """Download every month between start and end (inclusive). Safe to re-run: done months are skipped."""
    for month, url in list_monthly_files().items():
        if start <= month <= end:
            try:
                download_month(month, url)
            except Exception as exc:  # keep going, one bad month should not stop the rest
                print(f"{month}: failed ({exc})")


if __name__ == "__main__":
    main(*sys.argv[1:])
