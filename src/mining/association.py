# src/mining/association.py

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def discretize_data(df):
    df = df.copy()

    # Rời rạc hóa absences
    df['absence_level'] = pd.cut(
        df['absences'],
        bins=[-1,0,5,15,100],
        labels=['0','low','medium','high']
    )

    # studytime giữ nguyên dạng mức
    df['studytime_level'] = df['studytime'].astype(str)

    # failures
    df['failures_level'] = df['failures'].astype(str)

    # chỉ lấy các cột cần
    df_ap = df[['absence_level','studytime_level','failures_level','pass']]

    # one-hot encoding
    df_ap = pd.get_dummies(df_ap.astype(str))

    return df_ap


def run_apriori(df, min_support=0.05, min_confidence=0.6):

    freq = apriori(df, min_support=min_support, use_colnames=True)

    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)

    rules = rules.sort_values(by=['confidence','lift'], ascending=False)

    return rules