# scripts/run_pipeline.py

from src.data.loader import load_data
from src.data.cleaner import clean_data
from src.features.builder import create_target, encode_categorical, get_feature_matrix

from src.mining.association import discretize_data, run_apriori
from src.mining.clustering import run_kmeans, cluster_profiling

from src.models.supervised import train_models
from src.models.semi_supervised import run_semi_supervised

from src.evaluation.plots import *
from src.utils.config import load_config

import pandas as pd
import os


def main():
    # =====================
    # LOAD CONFIG
    # =====================
    config = load_config()

    # =====================
    # LOAD DATA
    # =====================
    print("1) Load data")
    df = load_data(config["data_path"])
    print("  -> raw shape:", df.shape)

    # =====================
    # CLEAN
    # =====================
    print("2) Clean data")
    df = clean_data(df)
    print("  -> after clean shape:", df.shape)

    # =====================
    # PLOTS
    # =====================
    os.makedirs("outputs/figures", exist_ok=True)

    print("Plotting EDA charts")

    plot_distribution(df)
    plot_absence_vs_grade(df)
    plot_studytime(df)
    plot_failures(df)
    plot_heatmap(df)

    print("  -> saved EDA plots")

    # =====================
    # FEATURE
    # =====================
    print("3) Create target (pass/fail)")
    df = create_target(df, threshold=10)
    print("  -> value counts (pass):")
    print(df['pass'].value_counts().to_dict())

    print("4) Encode categorical features")
    df_enc = encode_categorical(df)

    os.makedirs("outputs", exist_ok=True)

    df_enc.head(5).to_csv("outputs/processed_head.csv", index=False)
    df_enc.to_csv("outputs/processed_full.csv", index=False)

    print("  -> saved processed data")

    print("5) Build X, y")
    X, y = get_feature_matrix(df_enc, target_col='pass', keep_grades=False)
    print("  -> X shape:", X.shape, " y shape:", y.shape)

    # =====================
    # APRIORI
    # =====================
    print("6) Running Apriori")

    df_ap = discretize_data(df)

    rules = run_apriori(
        df_ap,
        min_support=config["min_support"],
        min_confidence=config["min_confidence"]
    )

    rules.head(10).to_csv("outputs/apriori_rules.csv", index=False)

    print("  -> saved outputs/apriori_rules.csv")

    # =====================
    # CLUSTERING
    # =====================
    print("7) Running Clustering")

    labels = run_kmeans(
        X,
        k=config["k_clusters"],
        random_state=config["random_state"]
    )

    profile = cluster_profiling(X, labels)
    profile.to_csv("outputs/cluster_profile.csv")

    print("  -> saved outputs/cluster_profile.csv")

    # =====================
    # CLASSIFICATION
    # =====================
    print("8) Running Classification")

    results = train_models(X, y)

    plot_confusion_matrix(results["rf_cm"])

    pd.DataFrame([results]).to_csv("outputs/classification_results.csv", index=False)

    print("  -> F1 Logistic:", results["lr_f1"])
    print("  -> F1 RandomForest:", results["rf_f1"])

    # =====================
    # SEMI-SUPERVISED
    # =====================
    print("9) Running Semi-Supervised")

    f1_10 = run_semi_supervised(X, y, label_rate=0.1)
    f1_20 = run_semi_supervised(X, y, label_rate=0.2)

    print("  -> F1 Logistic:", results["lr_f1"])
    print("  -> F1 RandomForest:", results["rf_f1"])
    
    print("  -> PR-AUC Logistic:", results["lr_pr_auc"])
    print("  -> PR-AUC RandomForest:", results["rf_pr_auc"])


if __name__ == "__main__":
    main()