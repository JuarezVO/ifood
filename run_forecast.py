"""Funções de inferência — requer modelos treinados em memória (use run_train.py)."""

from src.forecast import inference_dt, inference_kmeans
from src.schemas import UserOffer, UserProfile

__all__ = [
    "UserProfile",
    "UserOffer",
    "inference_kmeans",
    "inference_dt",
]

if __name__ == "__main__":
    print(
        "Os modelos não são persistidos em disco.\n"
        "Execute run_train.py para treinar e inferir na mesma sessão,\n"
        "ou importe inference_kmeans / inference_dt passando os pipelines em memória."
    )
