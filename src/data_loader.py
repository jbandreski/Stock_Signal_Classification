import yfinance as yf
import pandas as pd

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df
