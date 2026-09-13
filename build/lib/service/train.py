"""Expanding-window demand forecasting with scikit-learn and no future-target features."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ("lag_1", "lag_7", "rolling_7", "price", "promotion", "weekday_sin", "weekday_cos")


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"date", "units", "price", "promotion"}.issubset(reader.fieldnames or []):
            raise ValueError("Expected date, units, price, promotion columns")
        try:
            rows = [{"date": date.fromisoformat(row["date"]), "units": float(row["units"]),
                     "price": float(row["price"]), "promotion": int(row["promotion"])} for row in reader]
        except ValueError as exc:
            raise ValueError("Invalid date or numeric value") from exc
    rows.sort(key=lambda row: row["date"])
    if len(rows) < 35 or len({r["date"] for r in rows}) != len(rows):
        raise ValueError("At least 35 distinct daily observations are required")
    if any(b["date"] - a["date"] != timedelta(days=1) for a, b in zip(rows, rows[1:])):
        raise ValueError("Missing dates; impute explicitly before training")
    if any(r["units"] < 0 or r["price"] < 0 or r["promotion"] not in (0, 1) or
           not np.isfinite([r["units"], r["price"]]).all() for r in rows):
        raise ValueError("Invalid observation")
    return rows


def features(rows: list[dict], index: int) -> list[float]:
    if index < 7:
        raise ValueError("Seven prior days are required")
    day = rows[index]["date"].weekday()
    return [rows[index - 1]["units"], rows[index - 7]["units"],
            sum(r["units"] for r in rows[index - 7:index]) / 7,
            rows[index]["price"], rows[index]["promotion"],
            float(np.sin(2 * np.pi * day / 7)), float(np.cos(2 * np.pi * day / 7))]


def fit(rows: list[dict], end: int):
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
    model.fit([features(rows, i) for i in range(7, end)], [rows[i]["units"] for i in range(7, end)])
    return model


def train(path: Path, output: Path, refit_every: int = 7) -> dict:
    rows = load(path)
    if refit_every < 1:
        raise ValueError("refit_every must be positive")
    initial = max(28, int(len(rows) * .8))
    forecasts = []
    model = None
    for index in range(initial, len(rows)):
        if model is None or (index - initial) % refit_every == 0:
            model = fit(rows, index)
        predicted = max(0.0, float(model.predict([features(rows, index)])[0]))
        forecasts.append({"date": rows[index]["date"].isoformat(), "actual": rows[index]["units"],
                          "predicted": predicted})
    absolute_error = sum(abs(row["actual"] - row["predicted"]) for row in forecasts)
    actual_sum = sum(row["actual"] for row in forecasts)
    report = {"method": "expanding_window_one_step", "holdout_start": rows[initial]["date"].isoformat(),
              "holdout_days": len(forecasts), "refit_every_days": refit_every,
              "mae": absolute_error / len(forecasts),
              "wape": absolute_error / actual_sum if actual_sum else None}
    final_model = fit(rows, len(rows))
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "features": FEATURES, "last_date": rows[-1]["date"].isoformat(),
                 "history": [r["units"] for r in rows[-7:]]}, output / "model.joblib")
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path(os.getenv("SM_CHANNEL_TRAIN", "data")) / "demand.csv")
    parser.add_argument("--output", type=Path, default=Path(os.getenv("SM_MODEL_DIR", "artifacts")))
    args = parser.parse_args()
    print(json.dumps(train(args.train_file, args.output), indent=2))


if __name__ == "__main__":
    main()
