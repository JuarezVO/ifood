import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import joblib

from src.config import CLUSTERING_COLS_PROFILE

def plot_clusters(data: pd.DataFrame, clusters: np.ndarray) -> None:
    df = data.copy()
    df['cluster'] = clusters

    numeric_cols = df.select_dtypes(include='number').columns.drop('cluster').tolist()
    unique_clusters = sorted(df['cluster'].unique())
    n_cols = len(numeric_cols)
    n_rows = len(unique_clusters)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))

    for row, cluster in enumerate(unique_clusters):
        subset = df[df['cluster'] == cluster]
        for col_idx, col in enumerate(numeric_cols):
            ax = axes[row, col_idx]
            ax.hist(subset[col], bins=20, alpha=0.7)
            if row == 0:
                ax.set_title(col)
            if col_idx == 0:
                ax.set_ylabel(f'Cluster {cluster}')
            if col == 'gender':
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(['F', 'M', 'O'])

    for col_idx in range(n_cols):
        col = numeric_cols[col_idx]
        x_min = df[col].min()
        x_max = df[col].max()
        y_max = max(axes[row, col_idx].get_ylim()[1] for row in range(n_rows))
        for row in range(n_rows):
            axes[row, col_idx].set_xlim(x_min, x_max)
            axes[row, col_idx].set_ylim(0, y_max)

    plt.tight_layout()
    plt.savefig('data/images/clusters.png', dpi=300, bbox_inches='tight')

def elbow(data: pd.DataFrame) -> None:
    inertias = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.savefig('data/images/elbow.png', dpi=300, bbox_inches='tight')


def clustering(df: pd.DataFrame) -> pd.DataFrame:
    print('Clustering\n')
    data = df.groupby('account_id').agg(
        age=('age', 'first'),
        gender=('gender', 'first'),
        credit_card_limit=('credit_card_limit', 'first'),
        amount_medio=('amount', 'mean'),
        taxa_sucesso=('offer_success', 'mean'),
    ).reset_index()

    df_plot = data[CLUSTERING_COLS_PROFILE + ['taxa_sucesso']].dropna()
    df_model = df_plot[CLUSTERING_COLS_PROFILE].copy()

    encoders = {}
    for col in CLUSTERING_COLS_PROFILE:
        if df_model[col].dtype == 'object':
            encoders[col] = LabelEncoder().fit(df_model[col])
            df_model[col] = encoders[col].transform(df_model[col])
            df_plot[col] = df_model[col]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_model[CLUSTERING_COLS_PROFILE])
    # elbow(data_scaled)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    data['cluster'] = clusters
    df = df.merge(data[['account_id', 'cluster','taxa_sucesso']], on='account_id', how='left')

    joblib.dump(kmeans, 'model/kmeans.pkl')
    joblib.dump(scaler, 'model/kmeans_scaler.pkl')
    joblib.dump(encoders, 'model/kmeans_encoders.pkl')
    print('KMeans salvo em model/kmeans.pkl\n')

    plot_clusters(df_plot, clusters)
    return df
