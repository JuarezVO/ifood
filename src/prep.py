import pandas as pd
from src.config import PREP_AGE_MAX, PREP_COLS_TO_DROP, PREP_COLS_TO_RENAME, PREP_DROP_NA_SUBSETS, PREP_OFFER_CODE, TODAY

def _prep_transactions(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expande o campo 'value' para uma estrutura de colunas
    Args:
        transactions: DataFrame com as transações
    Returns:
        Tuple com os DataFrames de sucesso e falha
    """
    value_expanded = pd.json_normalize(transactions['value'])
    transactions = pd.concat([transactions.drop(columns='value'), value_expanded], axis=1)

    recebido = transactions.loc[transactions['event'] == 'offer received'].drop(['offer_id'],axis=1)
    visto = transactions.loc[transactions['event'] == 'offer viewed'].drop(['offer_id'],axis=1)
    completado = transactions.loc[transactions['event'] == 'offer completed'].drop(['offer id'],axis=1).rename(columns={'offer_id': 'offer id'})
    trans = transactions.loc[transactions['event'] == 'transaction'].drop(['offer_id','offer id'],axis=1)

    recebido_visto = recebido.merge(visto,on=['account_id', 'offer id'], suffixes=('_recv', '_view'), how='left')
    recebido_visto_completado = recebido_visto.merge(completado,on=['account_id', 'offer id'], suffixes=('_view', '_comp'), how='left')
    trans_com_oferta = trans.merge(recebido_visto_completado,on=['account_id'], suffixes=('', '_jornada'), how='left')

    # trans_com_oferta['view_offer'] = [0 if pd.isna(i) else 1 for i in trans_com_oferta['time_since_test_start_view']]
    # trans_com_oferta['completed_offer'] = [0 if pd.isna(i) else 1 for i in trans_com_oferta['time_since_test_start_jornada']]
    # trans_com_oferta['recv_offer'] = [0 if pd.isna(i) else 1 for i in trans_com_oferta['time_since_test_start_recv']]

    return trans_com_oferta


def prep_datasets(offers_path: str, profiles_path: str, transactions_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepara os datasets de offers, profiles e transactions
    Args:
        offers: DataFrame com as offers
        profiles: DataFrame com os profiles
        transactions: DataFrame com as transações
    Returns:
        Tuple com os DataFrames de offers, profiles e transactions
    """
    offers = pd.read_json(offers_path)

    profiles = pd.read_json(profiles_path)

    transactions = pd.read_json(transactions_path)
    transactions = _prep_transactions(transactions)

    df = transactions.merge(offers,left_on='offer id',right_on='id',how='left',suffixes=('_oferta', ''))
    df = df.merge(profiles,left_on='account_id',right_on='id',how='left',suffixes=('_perfil', ''))

    df['registered_on'] = pd.to_datetime(df['registered_on'],format='%Y%m%d')
    df['offer_success'] = [0 if pd.isna(i) else 1 for i in df['reward_jornada']]
    df['discount_per_minvalue'] = df['discount_value'] / df['min_value']
    df['channels'] = df['channels'].apply(lambda x: ''.join([c[0] for c in x]) if isinstance(x, list) else None)
    df['mean_amount_per_account'] = df.groupby('account_id')['amount'].transform('mean')
    df = df[df['age'] <= PREP_AGE_MAX]
    df['offer_code'] = df['offer id'].map(PREP_OFFER_CODE) # adicionar o tipo de oferta

    df = df.rename(columns=PREP_COLS_TO_RENAME).drop(columns=PREP_COLS_TO_DROP).dropna(subset=PREP_DROP_NA_SUBSETS)

    return df
