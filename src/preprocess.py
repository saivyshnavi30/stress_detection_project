import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna()

    # Remove outliers (MAD)
    median = np.median(df['HR'])
    mad = np.median(np.abs(df['HR'] - median))

    df = df[np.abs(df['HR'] - median) < 3 * mad]

    return df