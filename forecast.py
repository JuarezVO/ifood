import joblib
import pandas as pd
from pydantic import BaseModel

from src.config import CLUSTERING_COLS_PROFILE

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
    channels: str  # ex.: 'ems', 'we', 'wem', 'wems' (mesmo formato do prep)
    offer_type_bogo: int
    offer_type_discount: int


def inference_kmeans(user_profile: UserProfile) -> int:
    kmeans = joblib.load('model/kmeans.pkl')
    scaler = joblib.load('model/kmeans_scaler.pkl')
    encoders = joblib.load('model/kmeans_encoders.pkl')
    row = pd.DataFrame([{
        'age': user_profile.age,
        'gender': user_profile.gender,
        'credit_card_limit': user_profile.credit_card_limit,
        'amount_medio': user_profile.amount_medio,
    }], columns=CLUSTERING_COLS_PROFILE)
    for col, encoder in encoders.items():
        row[col] = encoder.transform(row[col])
    return int(kmeans.predict(scaler.transform(row))[0])

def inference_dt(user_profile: UserProfile, user_offer: UserOffer) -> int:
    """
    Infere sucesso da oferta (0/1) para usuários do cluster 1.
    O modelo foi treinado apenas nesse cluster (ver decision_tree.py).
    """
    model = joblib.load('model/decision_tree.pkl')
    row = pd.DataFrame([{
        'min_value': user_offer.min_value,
        'discount_value': user_offer.discount_value,
        'discount_per_minvalue': user_offer.discount_per_minvalue,
        'duration': user_offer.duration,
        'age': user_profile.age,
        'credit_card_limit': user_profile.credit_card_limit,
        'channels': user_offer.channels,
    }])
    row = pd.get_dummies(row, columns=['channels'], dtype=int)
    row['offer_type_bogo'] = user_offer.offer_type_bogo
    row['offer_type_discount'] = user_offer.offer_type_discount
    x = row.reindex(columns=model.feature_names_in_, fill_value=0)
    return int(model.predict(x)[0])

if __name__ == "__main__":
    user_profile_1 = UserProfile(
        age=20,
        gender='M',
        credit_card_limit=100,
        amount_medio=10,
        success_rate=0.1
    )

    user_profile_2 = UserProfile(
        age=30,
        gender='M',
        credit_card_limit=100,
        amount_medio=10,
        success_rate=0.1
    )

    user_profile = user_profile_2

    user_group = inference_kmeans(user_profile)
    print(f'cluster: {user_group}')

    if user_group == 1:
        user_offer = UserOffer(
            min_value=10.0,
            discount_value=2.0,
            discount_per_minvalue=0.2,
            duration=7,
            channels='em',
            offer_type_bogo=0,
            offer_type_discount=1,
        )
        print(f'Oferta poderá ser aceita: {inference_dt(user_profile, user_offer)}')
