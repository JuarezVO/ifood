import os
import warnings

# 1. Silencia o aviso do conflito de OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. Define manualmente o número de cores para o Loky (coloque o número de cores da sua CPU, ex: 4, 8, 16, ou apenas ignore)
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # Troque pelo número de threads da sua CPU se souber

# 3. Ignora os alertas visuais do Python na tela
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.config import (
    DECISION_TREE_TARGET_CLUSTER,
    OFFER_TYPE_EXCLUDE,
    PROCESSED_FULL_DATASET_PATH,
    RAW_OFFERS_PATH,
    RAW_PROFILE_PATH,
    RAW_TRANSACTIONS_PATH,
)
from src.prep import prep_datasets
from src.clustering import clustering
from src.describe_ds import describe_ds
from src.decision_tree import save_clustered_dataset, train_decision_tree
from src.forecast import inference_dt, inference_kmeans
from src.schemas import UserOffer, UserProfile
from src.spark_session import get_spark

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/images", exist_ok=True)

spark = get_spark()

try:
    df = prep_datasets(spark, RAW_OFFERS_PATH, RAW_PROFILE_PATH, RAW_TRANSACTIONS_PATH)

    describe_ds(df)

    df = df.filter(df["offer_type"] != OFFER_TYPE_EXCLUDE)
    dataset = df.toPandas()
    dataset.to_csv(PROCESSED_FULL_DATASET_PATH, index=False)

    df, kmeans_model, cluster_by_account = clustering(df)
    dt_model = train_decision_tree(dataset, cluster_by_account)
    save_clustered_dataset(dataset, cluster_by_account)

    print("Forecast (exemplo)\n")
    target_accounts = cluster_by_account[
        cluster_by_account["cluster"] == DECISION_TREE_TARGET_CLUSTER
    ]["account_id"]
    sample_id = target_accounts.iloc[0]
    sample_rows = dataset[dataset["account_id"] == sample_id]
    sample = sample_rows.iloc[0]
    user_profile = UserProfile(
        age=int(sample["age"]),
        gender=sample["gender"],
        credit_card_limit=float(sample["credit_card_limit"]),
        amount_medio=float(sample_rows["amount"].mean()),
        success_rate=float(sample_rows["offer_success"].mean()),
    )

    user_group = inference_kmeans(kmeans_model, user_profile)
    print(f"cluster: {user_group} (conta exemplo: {sample_id})")

    if user_group == DECISION_TREE_TARGET_CLUSTER:
        user_offer = UserOffer(
            min_value=10.0,
            discount_value=2.0,
            discount_per_minvalue=0.2,
            duration=7,
            channels="em",
            offer_type="discount",
        )
        proba = inference_dt(dt_model, user_profile, user_offer)
        print(f"Probabilidade da oferta ser aceita: {proba[1] * 100:.2f}%")
        print(f"Probabilidade da oferta não ser aceita: {proba[0] * 100:.2f}%")
finally:
    spark.stop()
