# Demand forecasting architecture

## Request and model path

Consecutive daily CSV → lag/calendar features → expanding-window one-step backtest → final Ridge model → joblib artifact → next-day API

## Boundaries

- **Input:** `demand.csv` with consecutive dates, nonnegative finite units and prices, and binary promotion flags. Price and promotion are assumed known at forecast time.
- **Runtime:** `MODEL_PATH` points to a reviewed artifact. `/forecast` requires the day after training and the artifact’s final seven actual unit observations.
- **Failure behavior:** Readiness returns 503 without the artifact; wrong dates or history return 422. Joblib deserialization is unsafe for untrusted files.

The FastAPI process exposes `/health/live` for process liveness and `/health/ready` for local prerequisites. Each HTTP response carries a generated `X-Request-ID`, `Cache-Control: no-store`, and `X-Content-Type-Options: nosniff`. JSON request logs record method, path, status, request ID, and duration, never request bodies, query strings, credentials, or user data. Logs are local process telemetry, not a claim of production monitoring.

The [single Helm chart](../k8s/helm/demand-forecast-service/) provides rolling updates, probes, resource bounds, security contexts, optional HPA and NetworkPolicy, and a PDB. [Argo CD](../k8s/argocd/application.yaml) points to that chart. Values need environment review before deployment, especially image pull access, ingress peers, external inference egress, and artifact mounts.

## Limits

The one-step backtest uses observed prior days as they become available. There is no multi-step forecast, external demand evaluation, or live SageMaker deployment.
