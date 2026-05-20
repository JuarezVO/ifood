import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import CLUSTERING_COLS_PROFILE, DECISION_TREE_FEATURES
from src.schemas import UserOffer, UserProfile


def inference_kmeans(kmeans_model: Pipeline, user_profile: UserProfile) -> int:
    row = pd.DataFrame([{
        "age": user_profile.age,
        "gender": user_profile.gender,
        "credit_card_limit": user_profile.credit_card_limit,
        "amount_medio": user_profile.amount_medio,
    }])
    return int(kmeans_model.predict(row[CLUSTERING_COLS_PROFILE])[0])


def inference_dt(
    dt_model: Pipeline,
    user_profile: UserProfile,
    user_offer: UserOffer,
) -> np.ndarray:
    row = pd.DataFrame([{
        "min_value": user_offer.min_value,
        "discount_value": user_offer.discount_value,
        "discount_per_minvalue": user_offer.discount_per_minvalue,
        "duration": user_offer.duration,
        "age": user_profile.age,
        "credit_card_limit": user_profile.credit_card_limit,
        "channels": user_offer.channels,
        "offer_type": user_offer.offer_type,
    }])
    proba = dt_model.predict_proba(row[DECISION_TREE_FEATURES])[0]
    classes = list(dt_model.named_steps["clf"].classes_)
    fail_idx = classes.index(0)
    success_idx = classes.index(1)
    return np.array([proba[fail_idx], proba[success_idx]])
