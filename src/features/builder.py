# src/features/builder.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def create_target(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    """Thêm cột 'pass' (1 nếu G3 >= threshold, ngược lại 0). Trả về df copy."""
    df = df.copy()
    df['pass'] = (df['G3'] >= threshold).astype(int)
    return df

def encode_categorical(df: pd.DataFrame, drop_cols=None) -> pd.DataFrame:
    """
    Mã hoá các cột object bằng LabelEncoder (đơn giản).
    drop_cols: danh sách cột không muốn mã hoá.
    Trả về dataframe mới.
    """
    df = df.copy()
    if drop_cols is None:
        drop_cols = []
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        if c in drop_cols:
            continue
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
    return df

def get_feature_matrix(df: pd.DataFrame, target_col: str = 'pass', keep_grades: bool = False):
    """
    Trả về X, y.
    Nếu keep_grades=False thì remove G1,G2,G3 để tránh leakage (dự đoán sớm).
    Nếu muốn baseline có G1/G2, set keep_grades=True.
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not in df")
    y = df[target_col].copy()
    drop_cols = [target_col]
    if not keep_grades:
        for g in ['G1','G2','G3']:
            if g in df.columns:
                drop_cols.append(g)
    X = df.drop(columns=drop_cols, errors='ignore')
    return X, y

def scale_features(X_train, X_test=None):
    """
    StandardScaler: nếu chỉ truyền X_train thì trả về (X_train_scaled, scaler).
    Nếu truyền X_test thì trả về (X_train_scaled, X_test_scaled, scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is None:
        return X_train_scaled, scaler
    else:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler