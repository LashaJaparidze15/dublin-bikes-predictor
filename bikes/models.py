"""Baselines and models for "how many bikes will station X have in N minutes?".

Every model follows the same small interface: fit(train_df) then predict(df) -> array,
so the notebook can compare them in one loop.
"""
from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bikes.config import MODELS_DIR
from bikes.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class PersistenceBaseline:
    """'Nothing changes': predict that the station will have exactly what it has now."""

    def fit(self, df):
        return self

    def predict(self, df):
        return df["bikes"].to_numpy(dtype=float)


class HistoricalAverageBaseline:
    """Average bike count for this station, weekday/weekend, and 10-minute slot of the day.

    Ignores the current state completely, so it shows how much the daily routine alone explains.
    """

    def fit(self, df):
        keys = pd.DataFrame(self._key_values(df))
        keys["y"] = df["target_bikes"].to_numpy()
        self.table_ = keys.groupby(["station_id", "weekend", "slot"])["y"].mean()
        self.fallback_ = keys.groupby("station_id")["y"].mean()
        return self

    def predict(self, df):
        idx = pd.MultiIndex.from_frame(pd.DataFrame(self._key_values(df)))
        pred = np.array(self.table_.reindex(idx), dtype=float)
        missing = np.isnan(pred)
        fallback = np.array(self.fallback_.reindex(df["station_id"].astype(int)), dtype=float)
        pred[missing] = fallback[missing]
        # a station that did not exist in the training period gets the network-wide average
        pred[np.isnan(pred)] = self.fallback_.mean()
        return pred

    @staticmethod
    def _key_values(df):
        local = df["target_time"].dt.tz_convert("Europe/Dublin")
        return {"station_id": df["station_id"].astype(int).to_numpy(),
                "weekend": (local.dt.day_of_week >= 5).to_numpy(),
                "slot": (local.dt.hour * 6 + local.dt.minute // 10).to_numpy()}


class RidgeModel:
    """Linear regression on the numeric features (station identity left out on purpose)."""

    def fit(self, df):
        self.pipe_ = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        X = df[NUMERIC_FEATURES].fillna(0)
        self.pipe_.fit(X, df["target_bikes"])
        return self

    def predict(self, df):
        return self.pipe_.predict(df[NUMERIC_FEATURES].fillna(0))


@dataclass
class LGBMRegressorModel:
    """Gradient boosted trees. Uses station_id as a native categorical feature.

    It predicts the CHANGE in bikes (target - bikes now) rather than the count itself. Most of
    the time a station barely changes in 30 minutes, so the persistence baseline is very strong;
    learning the change lets the model start from that and only learn the corrections.
    Huber loss keeps big rebalancing jumps (a van dropping 20 bikes) from dominating training.
    """
    params: dict = field(default_factory=lambda: dict(
        objective="huber", n_estimators=600, learning_rate=0.05, num_leaves=63,
        min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1))

    def fit(self, df):
        self.model_ = lgb.LGBMRegressor(**self.params)
        self.model_.fit(df[FEATURES], df["target_bikes"] - df["bikes"])
        return self

    def predict(self, df):
        # a station can never have fewer than 0 bikes or more bikes than docks in total
        upper = (df["bikes"] + df["docks"]).to_numpy()
        return np.clip(df["bikes"].to_numpy() + self.model_.predict(df[FEATURES]), 0, upper)


@dataclass
class LGBMStatusClassifier:
    """Three classes for the user-facing question: 0 = empty, 1 = OK, 2 = full."""
    params: dict = field(default_factory=lambda: dict(
        n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1))

    def fit(self, df):
        self.model_ = lgb.LGBMClassifier(**self.params)
        self.model_.fit(df[FEATURES], df["target_status"])
        return self

    def predict_proba(self, df):
        return self.model_.predict_proba(df[FEATURES])

    def predict(self, df):
        return self.predict_proba(df).argmax(axis=1)


def regression_scores(y_true, y_pred) -> dict:
    """MAE in bikes is the headline number: 'on average we are off by this many bikes'."""
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "within_2_bikes": float(np.mean(np.abs(np.asarray(y_true) - y_pred) <= 2))}


def empty_scores(is_empty_true, empty_proba, threshold: float = 0.5) -> dict:
    """Scores for the 'will it be empty?' question. Empty is the rare class, so accuracy is useless."""
    pred = (np.asarray(empty_proba) >= threshold).astype(int)
    return {"precision": precision_score(is_empty_true, pred, zero_division=0),
            "recall": recall_score(is_empty_true, pred, zero_division=0),
            "F1": f1_score(is_empty_true, pred, zero_division=0),
            "PR_AUC": average_precision_score(is_empty_true, empty_proba)}


def save_bundle(bundle: dict, name: str = "bundle.joblib") -> None:
    """Save everything the live predictor needs (models + station categories) in one file."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / name, compress=3)


def load_bundle(name: str = "bundle.joblib") -> dict:
    return joblib.load(MODELS_DIR / name)
