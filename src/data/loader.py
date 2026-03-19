# src/data/loader.py

import pandas as pd

def load_data(path=None):
    if path is None:
        path = "data/student-mat.csv"

    df = pd.read_csv(path, sep=";")
    return df