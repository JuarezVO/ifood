import datetime as dt

TODAY= dt.datetime.now()
TODAY_STR = TODAY.strftime('%Y-%m-%d')

PREP_COLS_TO_DROP = ['event_recv', 'event_view', 'event_jornada','amount_jornada', 'reward_view','amount_view','reward_recv', 'amount_view', 'reward_view', 'reward', 'id_perfil','id', 'amount_recv','event','registered_on','offer id']
PREP_COLS_TO_RENAME = {'reward_jornada': 'reward','time_since_test_start':'time_since_test_start_transaction'}
PREP_DROP_NA_SUBSETS = ['offer_code', 'account_id']
PREP_AGE_MAX = 80

PREP_OFFER_CODE = {'ae264e3637204a6fb9bb56bc8210ddfd': 'Offer 1', '0b1e1539f2cc45b7b9fa7c272da2e1d7': 'Offer 2',
 '2906b810c7d4411798c6938adc9daaa5': 'Offer 3', '2298d6c36e964ae4a3e7e9706d1fb8c2': 'Offer 4',
 '4d5c57ea9a6940dd891ad53e9dbe8da0': 'Offer 5', '5a8bc65990b245e5a138643cd4eb9837': 'Offer 6',
 '3f207df678b143eea3cee63160fa8bed': 'Offer 7', 'fafdcd668e3743c1bb461111dcafc2a4': 'Offer 8',
 '9b98b8c7a33c4b65b9aebfe6a799e6d9': 'Offer 9', 'f19421c1d4aa40978ebb69ca19b0e20d': 'Offer 10'}

CLUSTERING_COLS = ['amount','channels','offer_type', 'age', 'gender', 'credit_card_limit']
CLUSTERING_COLS_PROFILE = ['age', 'gender', 'credit_card_limit', 'amount_medio']
