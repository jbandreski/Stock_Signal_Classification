import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRANSACTION_COST = 0.001   # 0.1 % per trade (round-trip)
LABEL_TO_POS = {0: 0, 1: 1, 2: -1}  # Hold=0, Buy=+1, Sell=-1

def run_backtest(test_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Vectorised backtest incorporating transaction costs.
    Returns a DataFrame with daily strategy and benchmark returns.
    """
    df = test_df.copy()
    df["Pred"] = y_pred
    df["Position"] = df["Pred"].map(LABEL_TO_POS)

    # Actual next-day return (signal acts on close, fills at next close)
    df["Market_Return"] = df["Return"].shift(-1)

    # Transaction cost applies whenever position changes
    df["Trade"]  = df["Position"].diff().abs()
    df["Cost"]   = df["Trade"] * TRANSACTION_COST

    df["Strat_Return"] = df["Position"] * df["Market_Return"] - df["Cost"]
    df.dropna(inplace=True)

    return df

def financial_metrics(df: pd.DataFrame) -> dict:
    """Compute and print Sharpe ratio, max drawdown, cumulative return, win rate."""
    strat  = df["Strat_Return"]
    market = df["Market_Return"]

    cum_strat  = (1 + strat).cumprod()
    cum_market = (1 + market).cumprod()

    # Sharpe (annualised, 252 trading days)
    sharpe = (strat.mean() / (strat.std() + 1e-9)) * np.sqrt(252)

    # Maximum drawdown
    roll_max   = cum_strat.cummax()
    drawdown   = (cum_strat - roll_max) / roll_max
    max_dd     = drawdown.min()

    # Win rate (trades that made money)
    trades = strat[strat != 0]
    win_rate = (trades > 0).mean() if len(trades) > 0 else 0.0

    metrics = {
        "Cumulative Return": cum_strat.iloc[-1] - 1,
        "Sharpe Ratio":      sharpe,
        "Max Drawdown":      max_dd,
        "Win Rate":          win_rate,
    }
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(cum_strat.values,  label="Strategy")
    plt.plot(cum_market.values, label="Buy & Hold")
    plt.title("Cumulative Returns")
    plt.xlabel("Trading Days")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/cumulative_returns.png", dpi=150)
    plt.show()

    return metrics
