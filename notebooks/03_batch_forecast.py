# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Batch forecast (inference)
# MAGIC Loads the **champion** model from Unity Catalog and writes the next 7 days of forecasts to a table
# MAGIC that planners, dashboards and the app can read.

# COMMAND ----------
import sys, os, json
sys.path.append(os.path.abspath("../src"))
import mlflow
import numpy as np
import pandas as pd
from shelfsense.features import FEATURES, build_features, future_frame

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "shelfsense_dev")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
model_name = f"{catalog}.{schema}.shelfsense_forecaster"
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
uri = f"models:/{model_name}@champion"
model = mlflow.sklearn.load_model(uri)
info = mlflow.models.get_model_info(uri)
sku_codes = mlflow.artifacts.load_dict(f"runs:/{info.run_id}/sku_codes.json")

sales = spark.table(f"{catalog}.{schema}.sales_daily").toPandas()
sales["date"] = pd.to_datetime(sales["date"])
fut = future_frame(sales)
feats = build_features(pd.concat([sales, fut], ignore_index=True), sku_codes)
feats = feats[feats["date"] > sales["date"].max()].copy()
feats["forecast_units"] = np.clip(model.predict(feats[FEATURES]), 0, None).round()
feats["model_version"] = info.registered_model_version if hasattr(info, "registered_model_version") else None
feats["run_id"] = info.run_id
out = feats[["date", "sku", "product_name", "category", "forecast_units", "unit_price", "promo", "run_id"]]

(spark.createDataFrame(out).write.mode("append").option("mergeSchema", "true")
      .saveAsTable(f"{catalog}.{schema}.forecasts"))
display(out)
