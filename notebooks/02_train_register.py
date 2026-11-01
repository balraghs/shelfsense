# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Train, evaluate, track and register
# MAGIC Trains the forecaster, logs the run to MLflow, and registers the model in Unity Catalog.
# MAGIC The new model gets the alias **champion** only if it passes the quality gate and beats the current champion.

# COMMAND ----------
import sys, os
sys.path.append(os.path.abspath("../src"))
import mlflow
from mlflow.models import infer_signature
from mlflow import MlflowClient
from shelfsense.features import FEATURES, build_features
from shelfsense.model import MAX_WAPE, train_and_evaluate

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "shelfsense_dev")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
model_name = f"{catalog}.{schema}.shelfsense_forecaster"

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{user}/shelfsense-forecasting")

# COMMAND ----------
try:
    table_version = dbutils.jobs.taskValues.get(taskKey="ingest_validate", key="sales_table_version")
except Exception:
    table_version = None  # running interactively
sales = spark.table(f"{catalog}.{schema}.sales_daily").toPandas()

# COMMAND ----------
# Experiment: try a few settings and keep every run in MLflow.
candidates = [
    {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15},
    {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31},
    {"max_iter": 400, "learning_rate": 0.03, "max_leaf_nodes": 63},
]
best = None
for params in candidates:
    with mlflow.start_run(run_name=f"hgb-{params['max_leaf_nodes']}leaves") as run:
        result = train_and_evaluate(sales, params=params)
        mlflow.log_params(params)
        mlflow.log_param("sales_table", f"{catalog}.{schema}.sales_daily")
        mlflow.log_param("sales_table_version", table_version)
        mlflow.log_metrics({"wape": result.metrics["WAPE %"], "mape": result.metrics["MAPE %"],
                            "rmse": result.metrics["RMSE units"], "bias": result.metrics["Bias %"],
                            "baseline_wape": result.baseline_metrics["WAPE %"],
                            "passes_gate": int(result.passes_gate)})
        X = build_features(sales, result.sku_codes).dropna(subset=FEATURES)[FEATURES].tail(50)
        mlflow.sklearn.log_model(result.model, "model", signature=infer_signature(X, result.model.predict(X)),
                                 input_example=X.head(3))
        mlflow.log_dict(result.sku_codes, "sku_codes.json")
        if best is None or result.metrics["WAPE %"] < best[1].metrics["WAPE %"]:
            best = (run.info.run_id, result)

run_id, result = best
print(f"Best run {run_id}: WAPE {result.metrics['WAPE %']:.2f}% vs baseline {result.baseline_metrics['WAPE %']:.2f}%")

# COMMAND ----------
# Quality gate before registration.
if not result.passes_gate:
    raise ValueError(f"Model failed the gate (must beat baseline and WAPE < {MAX_WAPE}%). Not registering.")

mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
client = MlflowClient()
client.set_registered_model_alias(model_name, "challenger", mv.version)

# Promote to champion only if better than the current champion.
try:
    champ = client.get_model_version_by_alias(model_name, "champion")
    champ_wape = client.get_run(champ.run_id).data.metrics["wape"]
except Exception:
    champ, champ_wape = None, float("inf")

if result.metrics["WAPE %"] < champ_wape:
    client.set_registered_model_alias(model_name, "champion", mv.version)
    print(f"Version {mv.version} is the new champion (previous WAPE {champ_wape:.2f}).")
else:
    print(f"Version {mv.version} kept as challenger; champion v{champ.version} is still better.")
