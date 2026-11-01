# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest and validate sales data
# MAGIC Loads daily sales into a Unity Catalog table and stops the job if the data fails validation.
# MAGIC In a real business this reads from the ERP or distributor feed; here we generate synthetic data.

# COMMAND ----------
import sys, os
sys.path.append(os.path.abspath("../src"))
from shelfsense.data import generate_sales, prepare, validate_sales

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "shelfsense_dev")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------
raw = generate_sales(days=730)
problems = validate_sales(raw)
if problems:
    # Data quality gate: a failed check stops the pipeline before training.
    raise ValueError("Data validation failed: " + "; ".join(problems))

sales = prepare(raw)
(spark.createDataFrame(sales)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{catalog}.{schema}.sales_daily"))
print(f"Wrote {len(sales):,} rows to {catalog}.{schema}.sales_daily")

# COMMAND ----------
# MAGIC %md
# MAGIC Delta tables keep history, so every training run can point at an exact table version.
# MAGIC That version is logged with the model in the next notebook (dataset versioning / lineage).

# COMMAND ----------
version = spark.sql(f"DESCRIBE HISTORY {catalog}.{schema}.sales_daily LIMIT 1").collect()[0]["version"]
dbutils.jobs.taskValues.set(key="sales_table_version", value=int(version))
print("sales_daily version:", version)
