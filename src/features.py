import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"MACD": macd_line, "MACD_Signal": signal_line})

def engineer_features(df: pd.DataFrame,
                      horizon: int = 5,
                      threshold: float = 0.01) -> pd.DataFrame:
    """
    Build feature matrix and label vector from raw OHLCV data.

    Labels (eq. 2-3 in paper):
        Buy  if forward_return > threshold
        Sell if forward_return < -threshold
        Hold otherwise
    """
    close = df["Close"].squeeze()

    feat = pd.DataFrame(index=df.index)

    # Daily returns (eq. 4)
    feat["Return"] = close.pct_change()

    # Moving averages
    feat["SMA_10"] = close.rolling(10).mean()
    feat["SMA_50"] = close.rolling(50).mean()

    # Rolling volatility
    feat["Volatility"] = feat["Return"].rolling(10).std()

    # RSI
    feat["RSI"] = compute_rsi(close)

    # MACD
    macd_df = compute_macd(close)
    feat["MACD"] = macd_df["MACD"]
    feat["MACD_Signal"] = macd_df["MACD_Signal"]

    # Forward return for labeling (look-ahead — kept separate from features)
    forward_return = close.pct_change(horizon).shift(-horizon)

    # Labels: 0 = Hold, 1 = Buy, 2 = Sell
    label = pd.Series(0, index=df.index, name="Label")
    label[forward_return >  threshold] = 1
    label[forward_return < -threshold] = 2

    feat["Label"] = label

    # Drop rows with NaN from rolling windows or forward shift
    feat.dropna(inplace=True)

    return feat
