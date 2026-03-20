# src/mining/clustering.py

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd


def run_kmeans(X, k, random_state=42):

    print("Running KMeans with k =", k)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=random_state)
    labels = model.fit_predict(X_scaled)

    return labels

def cluster_profiling(X, labels):
    df_cluster = pd.DataFrame(X).copy()
    df_cluster['cluster'] = labels

    profile = df_cluster.groupby('cluster').mean()

    return profile