import joblib
from pydantic import BaseModel

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
    offer_type_bogo: int
    offer_type_discount: int


def inference_kmeans(user_profile: UserProfile) -> int:
    kmeans = joblib.load('model/kmeans.pkl')
    return kmeans.predict([[user_profile.age, user_profile.gender, user_profile.credit_card_limit, user_profile.amount_medio]])

def inference_dt(
    min_value: float,
    discount_value: float,
    discount_per_minvalue: float,
    duration: int,
    age: int,
    credit_card_limit: float,
    channels_ems: int,
    channels_we: int,
    channels_wem: int,
    channels_wems: int,
    offer_type_bogo: int,
    offer_type_discount: int) -> int:
    """
    Infere a taxa de sucesso de uma oferta
    Args:
        min_value: float com o valor mínimo da oferta
        discount_value: float com o valor do desconto
        discount_per_minvalue: float com o valor do desconto por valor mínimo
        duration: int com a duração da oferta
        age: int com a idade do usuário
        credit_card_limit: float com o limite do cartão de crédito
        channels_ems: int com o número de canais EMS
        channels_we: int com o número de canais WE
        channels_wem: int com o número de canais WEM
        channels_wems: int com o número de canais WEMs
        offer_type_bogo: int com o tipo de oferta BOGO
        offer_type_discount: int com o tipo de oferta DISCOUNT
    Returns:
        float com a taxa de sucesso da oferta
    """
    model = joblib.load('model/decision_tree.pkl')
    return model.predict([[min_value, discount_value, discount_per_minvalue, duration, age, credit_card_limit, channels_ems, channels_we, channels_wem, channels_wems, offer_type_bogo, offer_type_discount]])

if __name__ == "__main__":
    user_profile = UserProfile(
        age=30,
        gender='M',
        credit_card_limit=1000,
        amount_medio=100,
        success_rate=0.1
    )

    user_group = inference_kmeans(user_profile)
    print(user_group)

    # print(inference(**user_offer.model_dump()))
