"""Tests for the grid, lag and target logic, which every model depends on."""
import numpy as np
import pandas as pd
import pytest

from bikes.features import add_lags, add_targets, add_time_features, build_grid


def snapshots(minutes, bikes, station_id=1, renting=None):
    """Fake readings for one station at the given minutes past 08:00 UTC on a Monday."""
    start = pd.Timestamp("2025-03-03 08:00")
    n = len(minutes)
    return pd.DataFrame({
        "last_reported": [start + pd.Timedelta(minutes=m) for m in minutes],
        "station_id": station_id,
        "num_bikes_available": bikes,
        "num_docks_available": [20 - b for b in bikes],
        "is_installed": True,
        "is_renting": renting if renting is not None else [True] * n,
        "is_returning": True,
        "capacity": 20,
    })


def test_grid_keeps_latest_reading_in_a_slot():
    grid = build_grid(snapshots([0, 5], [3, 4]))
    assert len(grid) == 1
    assert grid["bikes"].iloc[0] == 4


def test_grid_fills_short_gaps_only():
    # readings at 08:00 and 09:00, nothing in between: 2 slots filled, 3 left empty
    grid = build_grid(snapshots([0, 60], [5, 9]))
    assert len(grid) == 4  # 08:00, 08:10, 08:20 (filled), 09:00
    assert list(grid["bikes"]) == [5, 5, 5, 9]


def test_grid_drops_closed_station_readings():
    grid = build_grid(snapshots([0, 10, 20], [5, 0, 6], renting=[True, False, True]))
    assert 0 not in grid["bikes"].tolist()


def test_lags_are_missing_across_a_gap():
    grid = build_grid(snapshots([0, 10, 20, 30, 100], [1, 2, 3, 4, 9]))
    grid = add_lags(grid)
    row_0830 = grid[grid["time"] == pd.Timestamp("2025-03-03 08:30", tz="UTC")].iloc[0]
    assert row_0830["bikes_lag10"] == 3 and row_0830["bikes_lag30"] == 1
    row_0940 = grid[grid["time"] == pd.Timestamp("2025-03-03 09:40", tz="UTC")].iloc[0]
    assert np.isnan(row_0940["bikes_lag10"])  # 09:30 was never observed


def test_targets_look_the_right_distance_ahead():
    grid = build_grid(snapshots([0, 10, 20, 30, 40], [5, 4, 3, 0, 1]))
    out = add_targets(grid, horizon=30)
    first = out.iloc[0]
    assert first["target_bikes"] == 0 and first["target_empty"] == 1 and first["target_status"] == 0
    assert out["target_bikes"].isna().sum() == 3  # last three rows have no future 30 min away


def test_time_features_use_irish_time():
    summer = pd.DataFrame({"time": pd.to_datetime(["2025-07-01 07:00"]).tz_localize("UTC")})
    assert add_time_features(summer)["hour"].iloc[0] == pytest.approx(8.0)  # IST = UTC + 1
