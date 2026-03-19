# src/data/cleaner.py
def clean_data(df):
    # loại bản sao, thiết lập index lại
    df = df.drop_duplicates().reset_index(drop=True)
    return df