from src.prep import prep_datasets
from src.clustering import clustering
from src.describe_ds import describe_ds
from src.decision_tree import train_decision_tree

df = prep_datasets("data/raw/offers.json", "data/raw/profile.json", "data/raw/transactions.json")
df.to_csv("data/processed/full_dataset.csv", index=False)

describe_ds(df)

df = df[df['offer_type'] != 'informational']
df = clustering(df)
df.to_csv("data/processed/full_dataset_clustered.csv", index=False)

train_decision_tree(df)
