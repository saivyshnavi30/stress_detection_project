import pandas as pd

def extract_features(df):
    features = pd.DataFrame()

    features['HR'] = df['HR']
    features['Game_Intensity'] = df['Game_Intensity']

    # HRV calculation
    features['HR_diff'] = df['HR'].diff()
    features['HRV'] = features['HR_diff'].rolling(window=2).std()

    features = features.dropna()

    return features


def label_stress(df):
    def label(hr):
        if hr < 80:
            return 0   # Low
        elif hr < 95:
            return 1   # Medium
        else:
            return 2   # High

    return df['HR'].apply(label)