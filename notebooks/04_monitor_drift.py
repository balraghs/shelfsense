# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Monitor accuracy and drift
# MAGIC Two checks a product owner cares about:
# MAGIC 1. **Accuracy**: how wrong were last week's forecasts, now that actual sales are in?
# MAGIC 2. **Drift**: does recent demand look different from the data the model learned from?
# MAGIC If either breaches its threshold, the job flags that retraining (and a human review) is needed.

# COMMAND ----------
import sys, os
sys.path.append(os.path.abspath("../src"))
import numpy as np
import pandas as pd
from shelfsense.metrics import wape

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "shelfsense_dev")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")

WAPE_ALERT = 20.0   # same threshold as the release gate
PSI_ALERT = 0.2     # common rule of thumb: >0.2 means a significant shift

# COMMAND ----------
sales = spark.table(f"{catalog}.{schema}.sales_daily").toPandas()
sales["date"] = pd.to_datetime(sales["date"])
try:
    fc = spark.table(f"{catalog}.{schema}.forecasts").toPandas()
    fc["date"] = pd.to_datetime(fc["date"])
    joined = fc.merge(sales[["date", "sku", "units_sold"]], on=["date", "sku"])
    live_wape = wape(joined["units_sold"], joined["forecast_units"]) if len(joined) else None
except Exception:
    live_wape = None
print("Live WAPE on matured forecasts:", live_wape)

# COMMAND ----------
def psi(expected, actual, bins=10):
    """Population stability index between two samples."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))

end = sales["date"].max()
recent = sales[sales["date"] > end - pd.Timedelta(days=28)]
reference = sales[sales["date"] <= end - pd.Timedelta(days=28)]
drift = (pd.DataFrame([{"sku": s, "psi": psi(reference[reference.sku == s]["units_sold"],
                                             recent[recent.sku == s]["units_sold"])}
                       for s in sales["sku"].unique()]).sort_values("psi", ascending=False))
display(drift)

# COMMAND ----------
needs_retrain = (live_wape is not None and live_wape > WAPE_ALERT) or (drift["psi"] > PSI_ALERT).any()
dbutils.jobs.taskValues.set(key="needs_retrain", value=bool(needs_retrain))
print("Retraining recommended:", needs_retrain)
