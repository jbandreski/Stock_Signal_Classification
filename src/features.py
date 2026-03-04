import pandas as pd
import numpy as np

def add_features(df):

    df["returns"] = df["Close"].pct_change()

    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()

    df["volatility"] = df["returns"].rolling(10).std()

    df = df.dropna()

    return df


def create_labels(df, threshold=0.01):

    future_return = df["Close"].pct_change().shift(-1)

    conditions = [
        future_return > threshold,
        future_return < -threshold
    ]

    choices = [1, -1]

    df["signal"] = np.select(conditions, choices, default=0)

    return df
