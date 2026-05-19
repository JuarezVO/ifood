from pyspark.sql import SparkSession


def get_spark(app_name: str = "offer-forecast") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
