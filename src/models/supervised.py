# src/models/supervised.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc
)
import pandas as pd


def train_models(X, y):

    # =====================
    # SPLIT DATA
    # =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =====================
    # MODELS
    # =====================
    lr = LogisticRegression(max_iter=2000)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # =====================
    # PREDICT
    # =====================
    pred_lr = lr.predict(X_test)
    pred_rf = rf.predict(X_test)

    # PROBA (cho PR-AUC)
    prob_lr = lr.predict_proba(X_test)[:, 1]
    prob_rf = rf.predict_proba(X_test)[:, 1]

    # =====================
    # F1
    # =====================
    f1_lr = f1_score(y_test, pred_lr)
    f1_rf = f1_score(y_test, pred_rf)

    # =====================
    # PR-AUC
    # =====================
    precision_lr, recall_lr, _ = precision_recall_curve(y_test, prob_lr)
    pr_auc_lr = auc(recall_lr, precision_lr)

    precision_rf, recall_rf, _ = precision_recall_curve(y_test, prob_rf)
    pr_auc_rf = auc(recall_rf, precision_rf)

    # =====================
    # REPORT + CM
    # =====================
    report_lr = classification_report(y_test, pred_lr, output_dict=True)
    report_rf = classification_report(y_test, pred_rf, output_dict=True)

    cm_lr = confusion_matrix(y_test, pred_lr)
    cm_rf = confusion_matrix(y_test, pred_rf)

    # =====================
    # RESULT
    # =====================
    results = {
        "lr_f1": f1_lr,
        "rf_f1": f1_rf,
        "lr_pr_auc": pr_auc_lr,
        "rf_pr_auc": pr_auc_rf,
        "lr_cm": cm_lr.tolist(),
        "rf_cm": cm_rf.tolist()
    }

    return results