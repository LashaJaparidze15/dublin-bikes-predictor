"""One collection run: store the live feed, then predict and log the predictions.

Runs every 5 minutes from GitHub Actions (.github/workflows/collect.yml), or by hand:
    python -m bikes.collector
"""
import logging

from bikes import store
from bikes.config import MODELS_DIR
from bikes.live import fetch_stations, fetch_status
from bikes.predict import predict_now
from bikes.weather import fetch_forecast

log = logging.getLogger("collector")


def run() -> None:
    status = fetch_status()
    store.save_stations(fetch_stations())
    store.save_snapshots(status)
    log.info("stored %d station readings", len(status))

    if not (MODELS_DIR / "bundle.joblib").exists():
        log.info("no model bundle yet, skipping predictions")
        return
    try:
        preds = predict_now(store.recent_snapshots(minutes=80), fetch_forecast())
        if preds.empty:
            log.info("no stations renting right now (scheme closed overnight), nothing to predict")
            return
        store.save_predictions(preds)
        log.info("logged %d predictions", len(preds))
    except Exception:  # collection must still count as a success if prediction fails
        log.exception("prediction step failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
