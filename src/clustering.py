import matplotlib.pyplot as plt
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, col, first

from src.config import (
    CLUSTERING_COLS_PROFILE,
    GENDER_LABELS,
    IMAGE_CLUSTERS_PATH,
    KMEANS_N_CLUSTERS,
    MODEL_KMEANS_PIPELINE_PATH,
    PLOT_DPI,
    RANDOM_STATE,
)


def plot_clusters(data: pd.DataFrame, clusters: pd.Series) -> None:
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
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(GENDER_LABELS)

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


def clustering(df: DataFrame) -> DataFrame:
    print("Clustering\n")
    profile_df = df.groupBy("account_id").agg(
        first("age").alias("age"),
        first("gender").alias("gender"),
        first("credit_card_limit").alias("credit_card_limit"),
        avg("amount").alias("amount_medio"),
        avg("offer_success").alias("taxa_sucesso"),
    )

    model_df = profile_df.select("account_id", *CLUSTERING_COLS_PROFILE).dropna()

    indexer = StringIndexer(
        inputCol="gender",
        outputCol="gender_idx",
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=["age", "gender_idx", "credit_card_limit", "amount_medio"],
        outputCol="features_raw",
    )
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True,
    )
    kmeans = KMeans(
        k=KMEANS_N_CLUSTERS,
        seed=RANDOM_STATE,
        featuresCol="features",
        predictionCol="cluster",
    )
    pipeline = Pipeline(stages=[indexer, assembler, scaler, kmeans])
    model = pipeline.fit(model_df)
    clustered_profiles = model.transform(model_df)

    profiles_with_cluster = profile_df.join(
        clustered_profiles.select("account_id", "cluster"),
        on="account_id",
        how="left",
    )
    df = df.join(
        profiles_with_cluster.select("account_id", "cluster", "taxa_sucesso"),
        on="account_id",
        how="left",
    )

    model.write().overwrite().save(MODEL_KMEANS_PIPELINE_PATH)
    print(f"KMeans pipeline salvo em {MODEL_KMEANS_PIPELINE_PATH}\n")

    plot_df = clustered_profiles.select(
        "age",
        col("gender_idx").alias("gender"),
        "credit_card_limit",
        "amount_medio",
        "cluster",
    ).toPandas()
    plot_clusters(plot_df.drop(columns=["cluster"]), plot_df["cluster"])

    return df
