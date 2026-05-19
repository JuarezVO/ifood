import numpy as np
from pydantic import BaseModel
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from src.config import (
    DECISION_TREE_TARGET_CLUSTER,
    MODEL_DECISION_TREE_PIPELINE_PATH,
    MODEL_KMEANS_PIPELINE_PATH,
)
from src.spark_session import get_spark


class UserProfile(BaseModel):
    age: int
    gender: str
    credit_card_limit: float
    amount_medio: float
    success_rate: float


class UserOffer(BaseModel):
    min_value: float
    discount_value: float
    discount_per_minvalue: float
    duration: int
    channels: str
    offer_type: str


def inference_kmeans(spark: SparkSession, user_profile: UserProfile) -> int:
    model = PipelineModel.load(MODEL_KMEANS_PIPELINE_PATH)
    row_df = spark.createDataFrame([{
        "age": user_profile.age,
        "gender": user_profile.gender,
        "credit_card_limit": user_profile.credit_card_limit,
        "amount_medio": user_profile.amount_medio,
    }])
    cluster = model.transform(row_df).select("cluster").collect()[0][0]
    return int(cluster)


def inference_dt(
    spark: SparkSession,
    user_profile: UserProfile,
    user_offer: UserOffer,
) -> np.ndarray:
    model = PipelineModel.load(MODEL_DECISION_TREE_PIPELINE_PATH)
    row_df = spark.createDataFrame([{
        "min_value": user_offer.min_value,
        "discount_value": user_offer.discount_value,
        "discount_per_minvalue": user_offer.discount_per_minvalue,
        "duration": user_offer.duration,
        "age": user_profile.age,
        "credit_card_limit": user_profile.credit_card_limit,
        "channels": user_offer.channels,
        "offer_type": user_offer.offer_type,
        "offer_success": 0,
        "weight": 1.0,
    }])
    result = model.transform(row_df).select("probability").collect()[0][0]
    return np.array([1.0 - result[1], result[1]])


if __name__ == "__main__":
    spark = get_spark()
    try:
        user_profile = UserProfile(
            age=30,
            gender="M",
            credit_card_limit=100,
            amount_medio=10,
            success_rate=0.1,
        )

        user_group = inference_kmeans(spark, user_profile)
        print(f"cluster: {user_group}")

        if user_group == DECISION_TREE_TARGET_CLUSTER:
            user_offer = UserOffer(
                min_value=10.0,
                discount_value=2.0,
                discount_per_minvalue=0.2,
                duration=7,
                channels="em",
                offer_type="discount",
            )

            proba = inference_dt(spark, user_profile, user_offer)
            print(f"Probabilidade da oferta ser aceita: {proba[1] * 100:.2f}%")
            print(f"Probabilidade da oferta não ser aceita: {proba[0] * 100:.2f}%")
    finally:
        spark.stop()
