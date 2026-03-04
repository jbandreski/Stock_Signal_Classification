import pandas as pd
import numpy as np

def backtest(df, predictions):

    df = df.copy()

    df["prediction"] = predictions - 1

    df["strategy_returns"] = df["prediction"].shift(1) * df["returns"]

    cumulative = (1 + df["strategy_returns"]).cumprod()

    return cumulative
