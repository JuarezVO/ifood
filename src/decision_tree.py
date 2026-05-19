import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

import joblib

from time import perf_counter

from src.config import (
    DECISION_TREE_FEATURES,
    DECISION_TREE_GRID_CV,
    DECISION_TREE_GRID_PARAMS,
    DECISION_TREE_GRID_SCORING,
    DECISION_TREE_TARGET_CLUSTER,
    DECISION_TREE_TARGET_COL,
    DECISION_TREE_TEST_SIZE,
    MODEL_DECISION_TREE_PATH,
    RANDOM_STATE,
)

ts = perf_counter()

# Carrega o dataset e seleciona as features
def train_decision_tree(df: pd.DataFrame) -> None:
    """
    Treina o modelo de Decision Tree
    Args:
        df: DataFrame com os dados
    Returns:
        None
    """
    print('Decision Tree\n')
    print('Tamanho dos clusters: \n', df['cluster'].value_counts(),'\n')
    df = df[DECISION_TREE_FEATURES]
    c1 = df[df['cluster'] == DECISION_TREE_TARGET_CLUSTER].drop(columns=['cluster'])

    # One-hot encoding: cria colunas binárias para cada categoria de cada feature categórica
    categorical_cols = [col for col in c1.columns if c1[col].dtype == 'object' and col != DECISION_TREE_TARGET_COL]
    c1_encoded = pd.get_dummies(c1, columns=categorical_cols, dtype=int)

    # Separa as features e alvo
    x = c1_encoded.drop(columns=[DECISION_TREE_TARGET_COL])
    y = c1_encoded[DECISION_TREE_TARGET_COL]

    # Separa os dados de treino e teste
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=DECISION_TREE_TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    train_data = pd.concat([x_train, y_train], axis=1)
    print("Dataset:\n", train_data[DECISION_TREE_TARGET_COL].value_counts(),'\n')

    # Ajusta o desbalanceamento das classes
    smote = SMOTE(random_state=RANDOM_STATE)
    x_train, y_train = smote.fit_resample(x_train, y_train)
    train_balanced = pd.concat([x_train, y_train], axis=1)
    print("Dataset balanceado:\n", train_balanced[DECISION_TREE_TARGET_COL].value_counts(),'\n')

    # Define os modelos
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)

    # Ajusta o modelo
    grid = GridSearchCV(
        tree, DECISION_TREE_GRID_PARAMS, cv=DECISION_TREE_GRID_CV, scoring=DECISION_TREE_GRID_SCORING,
    )
    grid.fit(x_train, y_train)

    # Apresenta os resultados
    best_tree = grid.best_estimator_
    print(grid.best_params_)
    for feature, importance in sorted(zip(x.columns, grid.best_estimator_.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance}")

    report = classification_report(y_test, best_tree.predict(x_test), output_dict=True)
    print(pd.DataFrame(report))

    # Taxa de conversão atual (sem modelo)
    baseline = c1[DECISION_TREE_TARGET_COL].mean()
    modelo = report['1']['precision']
    ganho = (modelo - baseline) / baseline * 100
    print(f'Base: {baseline*100.0:.2f}% | Modelo: {modelo*100.0:.2f}% | Melhoria estimada: {ganho:.2f}%\n')

    joblib.dump(best_tree, MODEL_DECISION_TREE_PATH)
    print(f'Modelo salvo em {MODEL_DECISION_TREE_PATH}\n')

    print(f'Time: {round((perf_counter() - ts)/60, 2)}min')
