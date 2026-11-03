"""Make live predictions with the saved model bundle."""
import pandas as pd

from bikes.features import build_grid, make_features
from bikes.models import load_bundle


def predict_now(recent: pd.DataFrame, forecast: pd.DataFrame, bundle: dict | None = None,
                max_age_minutes: int = 30) -> pd.DataFrame:
    """Predict every horizon for every station from the last ~80 minutes of readings.

    `recent` uses the history column names (num_bikes_available, ...). Returns one row per
    station per horizon. p_empty / p_full come from the 30-minute classifier only.
    """
    bundle = bundle or load_bundle()
    stations = bundle["stations"]
    grid = build_grid(recent) if not recent.empty else recent
    if grid.empty:  # scheme closed (roughly 00:30 to 05:00) or no recent readings
        return pd.DataFrame()
    latest_time = grid["time"].max()
    rows = []
    for horizon, model in bundle["regressors"].items():
        feats = make_features(grid, forecast, stations, horizon=horizon, with_targets=False)
        feats = feats.sort_values("time").groupby("station_id", observed=True).tail(1)
        feats = feats[feats["time"] >= latest_time - pd.Timedelta(minutes=max_age_minutes)]
        out = pd.DataFrame({
            "station_id": feats["station_id"].astype(int).to_numpy(),
            "horizon_min": horizon,
            "target_time": feats["target_time"].to_numpy(),
            "bikes_now": feats["bikes"].to_numpy(),
            "docks_now": feats["docks"].to_numpy(),
            "predicted_bikes": model.predict(feats).round(1),
            "p_empty": None, "p_full": None,
        })
        if horizon == 30 and "classifier" in bundle:
            proba = bundle["classifier"].predict_proba(feats)
            out["p_empty"], out["p_full"] = proba[:, 0].round(3), proba[:, 2].round(3)
        rows.append(out)
    preds = pd.concat(rows, ignore_index=True)
    preds["made_at"] = pd.Timestamp.now(tz="UTC").floor("s")
    preds["model_version"] = bundle.get("version", "unknown")
    return preds
