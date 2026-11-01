# ShelfSense: 7-day sales forecasting for a CPG business

Case-study app for the *DevOps & MLOps for Product Leaders* session (E&ICT Academy, IIT Kanpur).
It shows the full lifecycle of a data-driven product: application development, data engineering,
CI/CD, ML engineering and a business decision at the end.

| Feature | Where in the app |
|---|---|
| Sales dashboard: historical sales and KPIs | **Sales dashboard** |
| Upload or generate synthetic sales data (validated on upload) | **Data management** |
| Sales by product and category, weekday pattern, promo lift | **Sales analytics** |
| Predict sales for the next 7 days, with a promotion what-if | **Forecast** |
| Accuracy (WAPE, MAPE, RMSE, bias) vs a naive baseline, feature importance | **Model insights** |
| Actual vs predicted on a 28-day holdout | **Actual vs predicted** |

## Tool stack

| Need | Tool |
|---|---|
| AI coding assistant | Claude Code or GitHub Copilot for the repo; Databricks Assistant inside notebooks |
| Application | Python + Streamlit |
| Source control | Git + GitHub |
| CI/CD | GitHub Actions (`.github/workflows/ci.yml`, `deploy.yml`) |
| Automated testing | Pytest (unit, data-quality and model-quality tests) |
| ML development | Databricks Free Edition notebooks (`notebooks/`) on serverless compute |
| ML framework | scikit-learn `HistGradientBoostingRegressor` |
| Experiment tracking and registry | MLflow on Databricks, models registered in Unity Catalog |
| Deployment | Databricks Asset Bundle (`databricks.yml`) → Databricks Job + Databricks App |

## Run locally (5 minutes)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -v              # 17 tests, including the model release gate
streamlit run app.py   # opens http://localhost:8501
```

## Repository layout

```
app.py                     Streamlit UI (six pages)
src/shelfsense/data.py     synthetic generator, validation (data quality gate), KPIs
src/shelfsense/features.py lag / rolling / calendar features (all ≥7 days old: no leakage)
src/shelfsense/model.py    train, holdout evaluation, release gate, 7-day forecast
src/shelfsense/metrics.py  WAPE, MAPE, RMSE, bias
tests/                     pytest suite run by CI
notebooks/01..04           Databricks pipeline: ingest → train+register → batch forecast → monitor drift
databricks.yml             bundle: weekly training job + app, dev and prod targets
app.yaml                   Databricks Apps entry point
.github/workflows/         CI (lint, test, package) and CD (dev auto, prod with approval)
```

## The model in one paragraph

One global gradient-boosting model covers all SKUs. Every feature is at least 7 days old
(sales 7/14/21/28 days ago, rolling means, weekday, month, festive season, promotion, price), so
the model forecasts each of the next 7 days directly from known history. It is scored on the last
28 days, which it never saw, against a naive baseline ("same weekday last week").
**Release gate:** the model ships only if it beats the baseline and its WAPE is under 20%.

## Set up on Databricks Free Edition

1. Sign up at databricks.com/learn/free-edition.
2. In the workspace: **Settings → Developer → Access tokens** → generate a token.
3. In GitHub: **Settings → Secrets and variables → Actions** → add `DATABRICKS_HOST`
   (your workspace URL) and `DATABRICKS_TOKEN`.
4. In GitHub: **Settings → Environments** → create `dev` and `production`; add yourself as a
   required reviewer on `production`. This is the human approval step before prod.
5. Push to `main`. CI runs tests; on success, Deploy publishes the bundle to dev, then waits for approval for prod.
6. In Databricks, run the job `shelfsense-training-dev` once to create tables, MLflow runs and the registered model.

Or from your laptop: `databricks bundle deploy -t dev && databricks bundle run shelfsense_app -t dev`.

### Free Edition limits that matter here

- Serverless compute only, with a daily quota; no GPUs (not needed for this model).
- Up to 3 Databricks Apps per account; an app stops 24 hours after it starts or redeploys. Restart it before a demo.
- One workspace, so `dev` and `prod` are separate folders and names in the same workspace.
- Max 5 concurrent job tasks; limited model-serving endpoints. This project uses batch forecasting, so it does not need a serving endpoint.

Source: https://docs.databricks.com/aws/en/getting-started/free-edition-limitations

## Classroom exercises

1. **Break the build.** Change `MAX_WAPE` in `model.py` to `5.0`, open a pull request, and watch CI block the merge.
2. **Bad data.** Upload a CSV with a negative `units_sold` value on the Data management page. The upload is rejected.
3. **Drift.** Generate data with a different seed or fewer days and compare Model insights before and after.
4. **Business decision.** On Forecast, put two Snacks SKUs on promotion and read the change in units and revenue.

On Databricks the app trains in-session on the data it has (synthetic or uploaded). The production
path is the job: `03_batch_forecast` writes `forecasts` to Unity Catalog using the `champion` model.
