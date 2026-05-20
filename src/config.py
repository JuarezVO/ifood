import datetime as dt

# --- Datas ---
TODAY = dt.datetime.now()
TODAY_STR = TODAY.strftime('%Y-%m-%d')

# --- Caminhos: dados brutos ---
RAW_OFFERS_PATH = 'data/raw/offers.json'
RAW_PROFILE_PATH = 'data/raw/profile.json'
RAW_TRANSACTIONS_PATH = 'data/raw/transactions.json'

# --- Caminhos: dados processados ---
PROCESSED_FULL_DATASET_PATH = 'data/processed/full_dataset.csv'
PROCESSED_CLUSTERED_DATASET_PATH = 'data/processed/full_dataset_clustered.csv'
CLUSTER_BY_ACCOUNT_PATH = 'data/processed/cluster_by_account.csv'

# --- Caminhos: imagens ---
IMAGE_CLUSTERS_PATH = 'data/images/clusters.png'
IMAGE_ELBOW_PATH = 'data/images/elbow.png'
IMAGE_DESCRIBE_OFFER_PATH = 'data/images/describe_offer.png'
IMAGE_DESCRIBE_PROFILE_PATH = 'data/images/describe_profile.png'
PLOT_DPI = 300

# --- Prep ---
PREP_COLS_TO_DROP = ['event_recv', 'event_view', 'event_jornada','amount_jornada', 'reward_view','amount_view','reward_recv', 'amount_view', 'reward_view', 'reward', 'id_perfil','id', 'amount_recv','event','registered_on','offer id']
PREP_COLS_TO_RENAME = {'reward_comp_jornada': 'reward', 'time_since_test_start': 'time_since_test_start_transaction'}
PREP_DROP_NA_SUBSETS = ['offer_code', 'account_id']
PREP_AGE_MAX = 80

PREP_OFFER_CODE = {'ae264e3637204a6fb9bb56bc8210ddfd': 'Offer 1', '0b1e1539f2cc45b7b9fa7c272da2e1d7': 'Offer 2',
 '2906b810c7d4411798c6938adc9daaa5': 'Offer 3', '2298d6c36e964ae4a3e7e9706d1fb8c2': 'Offer 4',
 '4d5c57ea9a6940dd891ad53e9dbe8da0': 'Offer 5', '5a8bc65990b245e5a138643cd4eb9837': 'Offer 6',
 '3f207df678b143eea3cee63160fa8bed': 'Offer 7', 'fafdcd668e3743c1bb461111dcafc2a4': 'Offer 8',
 '9b98b8c7a33c4b65b9aebfe6a799e6d9': 'Offer 9', 'f19421c1d4aa40978ebb69ca19b0e20d': 'Offer 10'}

# --- Clustering ---
CLUSTERING_COLS = ['amount','channels','offer_type', 'age', 'gender', 'credit_card_limit']
CLUSTERING_COLS_PROFILE = ['age', 'gender', 'credit_card_limit', 'amount_medio']
RANDOM_STATE = 42
KMEANS_N_CLUSTERS = 3
KMEANS_ELBOW_K_MIN = 2
KMEANS_ELBOW_K_MAX = 10

# --- Pipeline de treino ---
OFFER_TYPE_EXCLUDE = 'informational'

# --- Decision tree ---
DECISION_TREE_TARGET_CLUSTER = 2
DECISION_TREE_FEATURES = [
    'channels', 'offer_type', 'min_value', 'discount_value',
    'discount_per_minvalue', 'duration', 'age', 'credit_card_limit',
]
DECISION_TREE_CATEGORICAL_FEATURES = ['channels', 'offer_type']
DECISION_TREE_NUMERIC_FEATURES = [
    'min_value', 'discount_value', 'discount_per_minvalue',
    'duration', 'age', 'credit_card_limit',
]
DECISION_TREE_TARGET_COL = 'offer_success'
DECISION_TREE_TEST_SIZE = 0.2
DECISION_TREE_GRID_PARAMS = {
    'max_depth': [7, 9, 12],
    'min_samples_split': [100, 150],
}
DECISION_TREE_GRID_CV = 5

# --- Describe datasets ---
DESCRIBE_OFFER_COLS = ['offer_code', 'min_value', 'offer_type', 'channels']
DESCRIBE_PROFILE_COLS = ['credit_card_limit', 'age', 'amount']
PLOT_COLOR_SUCCESS = 'forestgreen'
PLOT_COLOR_FAIL = 'firebrick'
