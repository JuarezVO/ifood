import os

from src.config import (
    OFFER_TYPE_EXCLUDE,
    PROCESSED_CLUSTERED_DATASET_PATH,
    PROCESSED_FULL_DATASET_PATH,
    RAW_OFFERS_PATH,
    RAW_PROFILE_PATH,
    RAW_TRANSACTIONS_PATH,
)
from src.prep import prep_datasets
from src.clustering import clustering
from src.describe_ds import describe_ds
from src.decision_tree import train_decision_tree
from src.spark_session import get_spark

os.makedirs("model", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/images", exist_ok=True)

spark = get_spark()

try:
    df = prep_datasets(spark, RAW_OFFERS_PATH, RAW_PROFILE_PATH, RAW_TRANSACTIONS_PATH)
    df.write.mode("overwrite").parquet(PROCESSED_FULL_DATASET_PATH)

    describe_ds(df)

    df = df.filter(df["offer_type"] != OFFER_TYPE_EXCLUDE)
    df = clustering(df)
    df.write.mode("overwrite").parquet(PROCESSED_CLUSTERED_DATASET_PATH)
    train_decision_tree(df)
finally:
    spark.stop()
