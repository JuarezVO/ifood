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
    channels: str
    offer_type: str
