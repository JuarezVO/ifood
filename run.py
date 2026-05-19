from src.prep import prep_datasets
from src.clustering import clustering
from src.describe_profile import describe_profile
from src.describe_offer import describe_offer

import os
import pandas as pd

if not os.path.exists("dataset/full_dataset.csv"):
    df = prep_datasets("dataset/offers.json", "dataset/profile.json", "dataset/transactions.json")
    df.to_csv("dataset/full_dataset.csv", index=False)
else:
    df = pd.read_csv("dataset/full_dataset.csv")

df = df[df['offer_type'] != 'informational']
print(df.columns)
print(df)
# describe_offer(df)
# describe_profile(df)
df = clustering(df)
df.to_csv("dataset/full_dataset_clustered.csv", index=False)
quit()
