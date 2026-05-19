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

df = prep_datasets(RAW_OFFERS_PATH, RAW_PROFILE_PATH, RAW_TRANSACTIONS_PATH)
df.to_csv(PROCESSED_FULL_DATASET_PATH, index=False)

describe_ds(df)

df = df[df['offer_type'] != OFFER_TYPE_EXCLUDE]
df = clustering(df)
df.to_csv(PROCESSED_CLUSTERED_DATASET_PATH, index=False)
train_decision_tree(df)
