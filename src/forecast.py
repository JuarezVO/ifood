import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import CLUSTERING_COLS_PROFILE, DECISION_TREE_FEATURES
from src.schemas import UserOffer, UserProfile


def inference_kmeans(
    kmeans_model: KMeans,
    scaler: StandardScaler,
    encoders: dict[str, LabelEncoder],
    user_profile: UserProfile,
) -> int:
    row = pd.DataFrame([{
        "age": user_profile.age,
        "gender": user_profile.gender,
        "credit_card_limit": user_profile.credit_card_limit,
        "amount_medio": user_profile.amount_medio,
    }])

    # Aplica os encoders nas colunas categóricas (mesmo processo do treino)
    for column, encoder in encoders.items():
        row[column] = encoder.transform(row[column])

    # Aplica o scaler (mesmo processo do treino)
    row_scaled = scaler.transform(row[CLUSTERING_COLS_PROFILE])

    return int(kmeans_model.predict(row_scaled)[0])


def inference_dt(
    dt_model,
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
