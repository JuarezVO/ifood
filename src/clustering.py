import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, col, first
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.config import (
    CLUSTERING_COLS_PROFILE,
    CLUSTER_BY_ACCOUNT_PATH,
    IMAGE_CLUSTERS_PATH,
    KMEANS_N_CLUSTERS,
    PLOT_DPI,
    RANDOM_STATE,
)


def _gender_categories_by_frequency(pdf: pd.DataFrame) -> list[list[str]]:
    ordered = (
        pdf["gender"]
        .value_counts()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return [ordered]


def plot_clusters(data: pd.DataFrame, clusters: pd.Series, gender_labels: list[str]) -> None:
    df = data.copy()
    df["cluster"] = clusters

    numeric_cols = df.select_dtypes(include="number").columns.drop("cluster").tolist()
    unique_clusters = sorted(df["cluster"].unique())
    n_cols = len(numeric_cols)
    n_rows = len(unique_clusters)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))

    for row, cluster in enumerate(unique_clusters):
        subset = df[df["cluster"] == cluster]
        for col_idx, column in enumerate(numeric_cols):
            ax = axes[row, col_idx]
            ax.hist(subset[column], bins=20, alpha=0.7)
            if row == 0:
                ax.set_title(column)
            if col_idx == 0:
                ax.set_ylabel(f"Cluster {cluster}")
            if column == "gender":
                ax.set_xticks(range(len(gender_labels)))
                ax.set_xticklabels(gender_labels)

    for col_idx in range(n_cols):
        column = numeric_cols[col_idx]
        x_min = df[column].min()
        x_max = df[column].max()
        y_max = max(axes[row, col_idx].get_ylim()[1] for row in range(n_rows))
        for row in range(n_rows):
            axes[row, col_idx].set_xlim(x_min, x_max)
            axes[row, col_idx].set_ylim(0, y_max)

    plt.tight_layout()
    plt.savefig(IMAGE_CLUSTERS_PATH, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def _build_kmeans_pipeline(gender_categories: list[list[str]]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("gender", OrdinalEncoder(categories=gender_categories), ["gender"]),
            ("numeric", StandardScaler(), ["age", "credit_card_limit", "amount_medio"]),
        ],
    )
    return Pipeline([
        ("prep", preprocessor),
        ("kmeans", KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")),
    ])


def clustering(df: DataFrame) -> tuple[DataFrame, Pipeline, pd.DataFrame]:
    print("Clustering\n")
    profile_df = df.groupBy("account_id").agg(
        first("age").alias("age"),
        first("gender").alias("gender"),
        first("credit_card_limit").alias("credit_card_limit"),
        avg("amount").alias("amount_medio"),
        avg("offer_success").alias("taxa_sucesso"),
    )

    model_pdf = (
        profile_df.select("account_id", *CLUSTERING_COLS_PROFILE)
        .dropna()
        .toPandas()
    )

    gender_categories = _gender_categories_by_frequency(model_pdf)
    gender_labels = gender_categories[0]
    pipeline = _build_kmeans_pipeline(gender_categories)
    pipeline.fit(model_pdf[CLUSTERING_COLS_PROFILE])
    model_pdf["cluster"] = pipeline.predict(model_pdf[CLUSTERING_COLS_PROFILE])

    cluster_by_account = model_pdf[["account_id", "cluster"]].copy()
    cluster_by_account.to_csv(CLUSTER_BY_ACCOUNT_PATH, index=False)

    spark = df.sparkSession
    clustered_profiles = spark.createDataFrame(
        cluster_by_account.astype({"cluster": "int"}),
    )

    profiles_with_cluster = profile_df.join(
        clustered_profiles,
        on="account_id",
        how="left",
    )
    df = df.join(
        profiles_with_cluster.select("account_id", "cluster", "taxa_sucesso"),
        on="account_id",
        how="left",
    )

    plot_df = model_pdf.copy()
    gender_encoder = pipeline.named_steps["prep"].named_transformers_["gender"]
    plot_df["gender"] = gender_encoder.transform(plot_df[["gender"]]).ravel()
    plot_clusters(
        plot_df.drop(columns=["cluster", "account_id"]),
        plot_df["cluster"],
        gender_labels,
    )

    return df, pipeline, cluster_by_account
