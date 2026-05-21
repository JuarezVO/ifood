import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, col, first
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    CLUSTERING_COLS_PROFILE,
    CLUSTER_BY_ACCOUNT_PATH,
    GENDER_LABELS,
    IMAGE_CLUSTERS_PATH,
    KMEANS_N_CLUSTERS,
    PLOT_DPI,
    RANDOM_STATE,
)


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


def clustering(df: DataFrame) -> tuple[DataFrame, KMeans, StandardScaler, dict, pd.DataFrame]:
    print("Clustering\n")

    profile_df = df.groupBy("account_id").agg(
        first("age").alias("age"),
        first("gender").alias("gender"),
        first("credit_card_limit").alias("credit_card_limit"),
        # CORREÇÃO 1: usar 'amount' diretamente com avg, igual ao sklearn (mean de 'amount' no groupby)
        # O mean_amount_per_account do prep não deve ser usado aqui pois foi calculado antes dos filtros
        avg("amount").alias("amount_medio"),
        avg("offer_success").alias("taxa_sucesso"),
    )

    model_pdf = (
        profile_df.select("account_id", *CLUSTERING_COLS_PROFILE, "taxa_sucesso")
        .dropna()
        .toPandas()
    )

    # CORREÇÃO 2: usar LabelEncoder com ordem alfabética, igual ao sklearn
    encoders = {}
    df_model = model_pdf[CLUSTERING_COLS_PROFILE].copy()
    df_plot = model_pdf[CLUSTERING_COLS_PROFILE + ["taxa_sucesso"]].copy()

    for column in CLUSTERING_COLS_PROFILE:
        if df_model[column].dtype == "object":
            encoders[column] = LabelEncoder().fit(df_model[column])
            df_model[column] = encoders[column].transform(df_model[column])
            df_plot[column] = df_model[column]
            gender_labels = list(encoders[column].classes_)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_model[CLUSTERING_COLS_PROFILE])

    kmeans = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    clusters = kmeans.fit_predict(data_scaled)

    model_pdf["cluster"] = clusters
    cluster_by_account = model_pdf[["account_id", "cluster"]].copy()
    cluster_by_account.to_csv(CLUSTER_BY_ACCOUNT_PATH, index=False)

    spark = df.sparkSession
    clustered_profiles = spark.createDataFrame(
        cluster_by_account.astype({"cluster": "int"}),
    )

    profile_df = profile_df.join(clustered_profiles, on="account_id", how="left")
    df = df.join(
        profile_df.select("account_id", "cluster", "taxa_sucesso"),
        on="account_id",
        how="left",
    )

    plot_clusters(df_plot, clusters, gender_labels)

    return df, kmeans, scaler, encoders, cluster_by_account
