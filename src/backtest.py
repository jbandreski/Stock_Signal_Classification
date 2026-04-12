import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, cum_strat.values,  label="Strategy")
    ax.plot(df.index, cum_market.values, label="Buy & Hold")

    # Add test period shading
    ax.axvspan(df.index[0], df.index[-1], alpha=0.05, color='blue', label="Test Period")

    # Add start and end date labels on x axis
    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date   = df.index[-1].strftime("%Y-%m-%d")

    ax.set_xticks([df.index[0], df.index[len(df)//2], df.index[-1]])
    ax.set_xticklabels([start_date,
                        df.index[len(df)//2].strftime("%Y-%m-%d"),
                        end_date], rotation=15)

    ax.set_title(f"Cumulative Returns (Test Period: {start_date} to {end_date})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/cumulative_returns.png", dpi=150)
    plt.close()

    return metrics
