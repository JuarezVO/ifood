import os
import sys

from pyspark.sql import SparkSession


def get_spark(app_name: str = "offer-forecast") -> SparkSession:
    python = sys.executable
    os.environ.setdefault("PYSPARK_PYTHON", python)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", python)

    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
