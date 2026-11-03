"""Build the datasets and train the production model bundle.

    python -m bikes.train            # evaluate on the hold-out months, then retrain on everything and save

The notebooks import build_dataset() and split() from here so the analysis and the deployed
model are trained on exactly the same data.
"""
from datetime import date

import pandas as pd

from bikes.config import DATA_DIR
from bikes.features import HORIZONS, build_grid, load_history, load_stations, make_features
from bikes.models import (HistoricalAverageBaseline, LGBMRegressorModel, LGBMStatusClassifier,
                          PersistenceBaseline, empty_scores, regression_scores, save_bundle)
from bikes.weather import load_hourly

TEST_START = "2026-04-01"      # last three months of history are never used for training choices
TRAIN_SAMPLE = 0.35            # neighbouring 10-minute rows are near-duplicates, a sample is enough


def load_grid() -> pd.DataFrame:
    """The 10-minute grid for all downloaded months, cached because it takes a while to build."""
    cache = DATA_DIR / "grid.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    grid = build_grid(load_history())
    grid.to_parquet(cache, index=False)
    return grid


def build_dataset(horizon: int = 30, grid: pd.DataFrame | None = None) -> pd.DataFrame:
    grid = load_grid() if grid is None else grid
    return make_features(grid, load_hourly(), load_stations(), horizon=horizon)


def split(df: pd.DataFrame, test_start: str = TEST_START):
    """Time-based split: everything before test_start trains, everything after tests."""
    cut = pd.Timestamp(test_start, tz="UTC")
    return df[df["time"] < cut], df[df["target_time"] >= cut]


def main() -> None:
    grid = load_grid()
    regressors, classifier = {}, None
    for horizon in HORIZONS:
        df = build_dataset(horizon, grid)
        train, test = split(df)
        sample = train.sample(frac=TRAIN_SAMPLE, random_state=42)
        model = LGBMRegressorModel().fit(sample)
        print(f"{horizon} min | LightGBM", regression_scores(test["target_bikes"], model.predict(test)),
              "| persistence", regression_scores(test["target_bikes"], PersistenceBaseline().predict(test)))
        regressors[horizon] = LGBMRegressorModel().fit(df.sample(frac=TRAIN_SAMPLE, random_state=42))
        if horizon == 30:
            clf = LGBMStatusClassifier().fit(sample)
            print("30 min empty |", empty_scores(test["target_empty"], clf.predict_proba(test)[:, 0]))
            classifier = LGBMStatusClassifier().fit(df.sample(frac=TRAIN_SAMPLE, random_state=42))

    save_bundle({"regressors": regressors, "classifier": classifier, "stations": load_stations(),
                 "version": f"lgbm-{date.today().isoformat()}"})
    print("saved models/bundle.joblib")


if __name__ == "__main__":
    main()
