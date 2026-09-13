import csv
from datetime import date, timedelta

from fastapi.testclient import TestClient

from service import app, train


def dataset(path):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "units", "price", "promotion"])
        for i in range(90):
            writer.writerow([(date(2025, 1, 1) + timedelta(days=i)).isoformat(),
                             40 + (i % 7) * 2 + (i % 9 == 0) * 5, 10 + i % 3, int(i % 9 == 0)])


def test_backtest_and_api(tmp_path, monkeypatch):
    path = tmp_path / "demand.csv"
    dataset(path)
    report = train.train(path, tmp_path / "model")
    assert report["holdout_days"] == 18
    assert report["wape"] < .3
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model" / "model.joblib"))
    app.artifact.cache_clear()
    client = TestClient(app.app)
    assert client.get("/health/ready").status_code == 200
    result = client.post("/forecast", json={"forecast_date": (date(2025, 1, 1) + timedelta(days=90)).isoformat(),
                                           "last_seven_units": [40, 42, 44, 46, 48, 50, 52],
                                           "price": 10, "promotion": 0})
    assert result.status_code == 200
    assert result.json()["predicted_units"] >= 0


def test_future_actuals_cannot_change_first_holdout_prediction(tmp_path):
    path = tmp_path / "demand.csv"
    dataset(path)
    rows = train.load(path)
    first = train.fit(rows, 72).predict([train.features(rows, 72)])[0]
    rows[-1]["units"] = 100000
    second = train.fit(rows, 72).predict([train.features(rows, 72)])[0]
    assert first == second
