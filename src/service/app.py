"""One-step demand forecast API."""

from __future__ import annotations

import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .train import FEATURES

app = FastAPI(title="Demand forecast", version="0.1.0")


class ForecastRequest(BaseModel):
    forecast_date: date
    last_seven_units: list[float] = Field(min_length=7, max_length=7)
    price: float = Field(ge=0, allow_inf_nan=False)
    promotion: int = Field(ge=0, le=1)


@lru_cache(maxsize=1)
def artifact() -> dict:
    path = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    if not path.is_file():
        raise FileNotFoundError(path)
    bundle = joblib.load(path)
    if tuple(bundle["features"]) != FEATURES:
        raise ValueError("Model feature schema mismatch")
    return bundle


@app.get("/health/live")
def live() -> dict:
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> dict:
    try:
        artifact()
    except (FileNotFoundError, ValueError, KeyError):
        raise HTTPException(503, "Model artifact unavailable")
    return {"status": "ready"}


@app.post("/forecast")
def forecast(request: ForecastRequest) -> dict:
    if any(not np.isfinite(value) or value < 0 for value in request.last_seven_units):
        raise HTTPException(422, "History must contain seven nonnegative finite values")
    try:
        bundle = artifact()
    except (FileNotFoundError, ValueError, KeyError):
        raise HTTPException(503, "Model artifact unavailable")
    if request.forecast_date != date.fromisoformat(bundle["last_date"]) + timedelta(days=1):
        raise HTTPException(422, "This model supports only the day after its training data")
    day = request.forecast_date.weekday()
    history = request.last_seven_units
    if history != bundle["history"]:
        raise HTTPException(422, "History must match the model's final seven observations")
    vector = [[history[-1], history[0], sum(history) / 7, request.price, request.promotion,
               np.sin(2 * np.pi * day / 7), np.cos(2 * np.pi * day / 7)]]
    return {"forecast_date": request.forecast_date.isoformat(),
            "predicted_units": max(0.0, float(bundle["model"].predict(vector)[0]))}
