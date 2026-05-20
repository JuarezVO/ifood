from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    DECISION_TREE_CATEGORICAL_FEATURES,
    DECISION_TREE_FEATURES,
    DECISION_TREE_GRID_CV,
    DECISION_TREE_GRID_PARAMS,
    DECISION_TREE_NUMERIC_FEATURES,
    DECISION_TREE_TARGET_CLUSTER,
    DECISION_TREE_TARGET_COL,
    DECISION_TREE_TEST_SIZE,
    PROCESSED_CLUSTERED_DATASET_PATH,
    RANDOM_STATE,
)

ts = perf_counter()


def _categories_by_frequency(series: pd.Series) -> list[str]:
    return series.value_counts().sort_values(ascending=False).index.tolist()


def _spark_sample_weights(y: pd.Series) -> np.ndarray:
    total = len(y)
    n_classes = y.nunique()
    counts = y.value_counts()
    weight_per_class = {
        label: total / (n_classes * counts[label]) for label in counts.index
    }
    return y.map(weight_per_class).to_numpy()


def _build_decision_tree_pipeline(
    cat_categories: list[list[str]] | None = None,
) -> Pipeline:
    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        categories=cat_categories if cat_categories else "auto",
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", DECISION_TREE_NUMERIC_FEATURES),
            ("cat", ohe, DECISION_TREE_CATEGORICAL_FEATURES),
        ],
    )
    classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
    return Pipeline([
        ("prep", preprocessor),
        ("clf", classifier),
    ])


def _feature_names(pipeline: Pipeline) -> list[str]:
    prep = pipeline.named_steps["prep"]
    cat_encoder = prep.named_transformers_["cat"]
    return (
        DECISION_TREE_NUMERIC_FEATURES
        + list(cat_encoder.get_feature_names_out(DECISION_TREE_CATEGORICAL_FEATURES))
    )


def train_decision_tree(
    dataset: pd.DataFrame,
    cluster_by_account: pd.DataFrame,
) -> Pipeline:
    print("Decision Tree\n")

    merged = dataset.merge(cluster_by_account, on="account_id", how="inner")
    cluster_pdf = merged[merged["cluster"] == DECISION_TREE_TARGET_CLUSTER]
    print(f"Linhas no cluster {DECISION_TREE_TARGET_CLUSTER}: {len(cluster_pdf)}")

    X = cluster_pdf[DECISION_TREE_FEATURES]
    y = cluster_pdf[DECISION_TREE_TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=DECISION_TREE_TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("Dataset treino:")
    print(y_train.value_counts().sort_index())

    cat_categories = [
        _categories_by_frequency(X_train[col])
        for col in DECISION_TREE_CATEGORICAL_FEATURES
    ]
    pipeline = _build_decision_tree_pipeline(cat_categories)
    param_grid = {
        "clf__max_depth": DECISION_TREE_GRID_PARAMS["max_depth"],
        "clf__min_samples_split": DECISION_TREE_GRID_PARAMS["min_samples_split"],
    }
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=DECISION_TREE_GRID_CV,
        scoring="accuracy",
        n_jobs=1,
    )
    sample_weight = _spark_sample_weights(y_train)
    grid_search.fit(X_train, y_train, clf__sample_weight=sample_weight)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia no teste: {accuracy:.4f}")

    feature_names = _feature_names(best_model)
    importances = best_model.named_steps["clf"].feature_importances_
    for feature, importance in sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{feature}: {importance:.4f}")

    baseline = y.mean()
    test_pdf = X_test.copy()
    test_pdf["prediction"] = y_pred
    test_pdf[DECISION_TREE_TARGET_COL] = y_test.values
    positive_preds = test_pdf[test_pdf["prediction"] == 1]
    total_positive_preds = len(positive_preds)
    true_positive = len(
        positive_preds[positive_preds[DECISION_TREE_TARGET_COL] == 1],
    )
    modelo = true_positive / total_positive_preds if total_positive_preds else 0.0
    ganho = (modelo - baseline) / baseline * 100 if baseline else 0.0
    print(
        f"Base: {baseline * 100.0:.2f}% | Modelo: {modelo * 100.0:.2f}% | "
        f"Melhoria estimada: {ganho:.2f}%\n"
    )
    print(f"Time: {round((perf_counter() - ts) / 60, 2)}min")

    return best_model


def save_clustered_dataset(
    dataset: pd.DataFrame,
    cluster_by_account: pd.DataFrame,
) -> None:
    dataset.merge(cluster_by_account, on="account_id", how="left").to_csv(
        PROCESSED_CLUSTERED_DATASET_PATH,
        index=False,
    )
