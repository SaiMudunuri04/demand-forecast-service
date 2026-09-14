# Demand forecasting

An independent service for scikit-learn time-series backtesting and SageMaker training. This repository contains executable source, tests,
a container, Helm release, Argo CD application, and a CI workflow that builds an
immutable GHCR image after tests pass. It is a reference implementation; it has not
been deployed to a user's AWS account or Kubernetes cluster.

## Run

```sh
python -m pip install -e '.[test]'
python -m pytest -q
uvicorn service.app:app --reload
```

`/health/live` checks the process. `/health/ready` checks required local resources.
Configure data, model artifacts, and inference endpoints before serving traffic.

## Delivery

The workflow tests pull requests, then builds/pushes an image to GHCR on `main` and
updates the Helm image tag to the tested commit. Argo CD follows the Helm chart.
Install `argocd/application.yaml` in a cluster with Argo CD, set environment-specific
Helm values, provide secrets through a cluster secret manager, and make the package
pullable by the cluster. Model/data volumes are configured through `volumes` and
`volumeMounts`; use `envFromSecretName` for credentials. The workflow does not
provision AWS or a cluster.

`.env.example` contains placeholders only. Never commit credentials or private data.

## Training data and model lifecycle

Input `demand.csv` requires consecutive daily `date,units,price,promotion` rows. Price and promotion are assumed known at forecast time. The backtest refits on past observations only and predicts one day at a time; no future actual is used in a feature. WAPE and MAE come from the holdout period.

```sh
python -m service.train --train-file /path/to/demand.csv --output artifacts
MODEL_PATH=artifacts/model.joblib uvicorn service.app:app --host 127.0.0.1
python -m pip install -e '.[aws]'
python scripts/launch_sagemaker.py --training-s3-uri s3://your-bucket/demand-training/
```

The SageMaker training channel must contain `demand.csv`. The API accepts the model artifact's final seven actual unit observations and forecasts only the day after the training series; it rejects a different history to avoid presenting a scenario as a continuation of the trained series. Mount a reviewed model artifact read-only at `/models` before deployment. There is no claimed business-data accuracy or live AWS training run.
