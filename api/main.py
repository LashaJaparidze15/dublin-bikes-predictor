"""Small JSON API over the same predictor the dashboard uses.

    uvicorn api.main:app --reload
    GET /predictions             every station, 10/30/60 minutes ahead
    GET /predictions/{station}   one station
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import FastAPI, HTTPException

from bikes import store
from bikes.live import fetch_status
from bikes.models import load_bundle
from bikes.predict import predict_now
from bikes.weather import fetch_forecast

app = FastAPI(title="Dublin Bikes availability predictor")
BUNDLE = load_bundle()
_cache = {"at": 0.0, "preds": None}


def current_predictions():
    """Predictions are recomputed at most every 5 minutes, the speed of the feed itself."""
    if time.time() - _cache["at"] > 300:
        recent = store.recent_snapshots(minutes=80)
        now = fetch_status().rename(columns={"bikes": "num_bikes_available", "docks": "num_docks_available"})
        recent = now if recent.empty else pd.concat([recent, now])
        _cache.update(at=time.time(), preds=predict_now(recent, fetch_forecast(), BUNDLE))
    return _cache["preds"]


@app.get("/predictions")
def all_predictions():
    return current_predictions().to_dict(orient="records")


@app.get("/predictions/{station_id}")
def station_predictions(station_id: int):
    preds = current_predictions()
    if preds.empty:
        raise HTTPException(503, "no stations are renting right now (the scheme closes overnight)")
    mine = preds[preds["station_id"] == station_id]
    if mine.empty:
        raise HTTPException(404, f"no live data for station {station_id}")
    return mine.to_dict(orient="records")
