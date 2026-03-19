# src/models/semi_supervised.py

import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import f1_score


def run_semi_supervised(X, y, label_rate=0.2, random_state=42):

    y = y.copy().astype(int)

    rng = np.random.RandomState(random_state)

    # mask unlabeled
    unlabeled_mask = rng.rand(len(y)) > label_rate

    y_semi = y.copy()
    y_semi[unlabeled_mask] = -1

    model = LabelSpreading()
    model.fit(X, y_semi)

    y_pred = model.transduction_

    # đánh giá trên toàn bộ dữ liệu thật
    f1 = f1_score(y, y_pred)

    return f1