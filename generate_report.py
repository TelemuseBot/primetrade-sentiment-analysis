"""
Trader Performance vs Bitcoin Market Sentiment - Analysis Script
Primetrade.ai Data Science Assignment

Author: Mayank (Neuralix Labs)

This script:
 1. Loads and cleans the Hyperliquid historical trader data and the
    Bitcoin Fear/Greed sentiment index.
 2. Merges both datasets on date.
 3. Computes performance metrics (PnL, win rate, avg trade size, leverage
    proxy, long/short bias) segmented by market sentiment.
 4. Produces account-level and symbol-level breakdowns.
 5. Saves all output tables (CSV) and charts (PNG) used in the PDF report.

Run: python3 analysis.py
Outputs land in ./results/ (tables) and ./charts/ (figures)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

pd.set_option("display.width", 140)

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results")
CHARTS_DIR = os.path.join(BASE, "charts")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENTIMENT_COLORS = {
    "Extreme Fear": "#8B0000",
    "Fear": "#E4572E",
    "Neutral": "#B0A99F",
    "Greed": "#4C9A2A",
    "Extreme Greed": "#0B6E4F",
}


# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------------------

def load_sentiment(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["classification"] = df["classification"].str.strip()
    return df[["date", "value", "classification"]]


def load_trades(path):
    df = pd.read_csv(path)

    # Parse the IST timestamp (dd-mm-yyyy HH:MM) into a proper datetime + date
    df["Timestamp IST"] = pd.to_datetime(df["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["date"] = df["Timestamp IST"].dt.floor("D")

    # Numeric coercion
    num_cols = ["Execution Price", "Size Tokens", "Size USD", "Start Position",
                "Closed PnL", "Fee"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "Size USD"])
    return df


def merge_datasets(trades, sentiment):
    merged = trades.merge(sentiment, on="date", how="inner")
    merged["classification"] = pd.Categorical(
        merged["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    return merged


# ---------------------------------------------------------------------------
# 2. METRIC HELPERS
# ---------------------------------------------------------------------------

def add_trade_flags(df):
    """Flag closing trades (the ones that actually realize PnL)."""
    df = df.copy()
    df["is_close"] = df["Direction"].astype(str).str.contains("Close", case=False, na=False)
    df["is_win"] = (df["is_close"]) & (df["Closed PnL"] > 0)
    df["is_loss"] = (df["is_close"]) & (df["Closed PnL"] < 0)
    return df


# ---------------------------------------------------------------------------
# 3. CORE ANALYSES
# ---------------------------------------------------------------------------

def sentiment_summary(df):
    """Headline performance table segmented by sentiment classification."""
    closes = df[df["is_close"]]

    g = df.groupby("classification", observed=True)
    gc = closes.groupby("classification", observed=True)

    summary = pd.DataFrame({
        "total_trades": g.size(),
        "total_volume_usd": g["Size USD"].sum(),
        "avg_trade_size_usd": g["Size USD"].mean(),
        "closing_trades": gc.size(),
        "total_closed_pnl": gc["Closed PnL"].sum(),
        "avg_closed_pnl": gc["Closed PnL"].mean(),
        "win_rate_pct": gc["is_win"].mean() * 100,
        "avg_win_usd": gc.apply(lambda x: x.loc[x["is_win"], "Closed PnL"].mean()),
        "avg_loss_usd": gc.apply(lambda x: x.loc[x["is_loss"], "Closed PnL"].mean()),
        "unique_traders": g["Account"].nunique(),
    })
    summary = summary.reindex(SENTIMENT_ORDER)
    summary["pnl_per_trade"] = summary["total_closed_pnl"] / summary["closing_trades"]
    return summary.round(2)


def daily_pnl_vs_sentiment(df):
    daily = df[df["is_close"]].groupby("date").agg(
        daily_pnl=("Closed PnL", "sum"),
        n_trades=("Closed PnL", "size"),
    ).reset_index()
    sent_daily = df.groupby("date", observed=True).agg(
        sentiment_value=("value", "first"),
        classification=("classification", "first"),
    ).reset_index()
    return daily.merge(sent_daily, on="date", how="left").sort_values("date")


def long_short_bias(df):
    tab = pd.crosstab(df["classification"], df["Side"], normalize="index") * 100
    return tab.round(2)


def symbol_breakdown(df, top_n=10):
    closes = df[df["is_close"]]
    g = closes.groupby(["Coin", "classification"], observed=True)["Closed PnL"].sum().reset_index()
    pivot = g.pivot(index="Coin", columns="classification", values="Closed PnL").fillna(0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False).head(top_n)
    return pivot.round(2)


def trader_leaderboard(df, top_n=10):
    closes = df[df["is_close"]]
    g = closes.groupby("Account").agg(
        total_pnl=("Closed PnL", "sum"),
        trades=("Closed PnL", "size"),
        win_rate=("is_win", "mean"),
    )
    g["win_rate"] = (g["win_rate"] * 100).round(2)
    g = g.sort_values("total_pnl", ascending=False)
    return pd.concat([g.head(top_n), g.tail(top_n)]).round(2)


def sentiment_correlation(daily):
    d = daily.dropna(subset=["sentiment_value", "daily_pnl"])
    corr = d["sentiment_value"].corr(d["daily_pnl"])
    return corr


def contrarian_check(summary):
    """Are traders net profitable specifically during Fear (buying the dip) vs Greed?"""
    fear_pnl = summary.loc[["Extreme Fear", "Fear"], "total_closed_pnl"].sum()
    greed_pnl = summary.loc[["Greed", "Extreme Greed"], "total_closed_pnl"].sum()
    fear_wr = summary.loc[["Extreme Fear", "Fear"], "avg_closed_pnl"].mean()
    greed_wr = summary.loc[["Greed", "Extreme Greed"], "avg_closed_pnl"].mean()
    return {
        "fear_total_pnl": round(fear_pnl, 2),
        "greed_total_pnl": round(greed_pnl, 2),
        "fear_avg_pnl_per_trade": round(fear_wr, 2),
        "greed_avg_pnl_per_trade": round(greed_wr, 2),
    }


# ---------------------------------------------------------------------------
# 4. CHARTS
# ---------------------------------------------------------------------------

def chart_total_pnl_by_sentiment(summary):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = summary["total_closed_pnl"]
    colors = [SENTIMENT_COLORS[i] for i in vals.index]
    ax.bar(vals.index, vals.values, color=colors)
    ax.set_title("Total Closed PnL by Market Sentiment")
    ax.set_ylabel("Total Closed PnL (USD)")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "01_total_pnl_by_sentiment.png"), dpi=150)
    plt.close(fig)


def chart_winrate_by_sentiment(summary):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = summary["win_rate_pct"]
    colors = [SENTIMENT_COLORS[i] for i in vals.index]
    ax.bar(vals.index, vals.values, color=colors)
    ax.set_title("Win Rate (%) by Market Sentiment")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, max(60, vals.max() + 10))
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "02_winrate_by_sentiment.png"), dpi=150)
    plt.close(fig)


def chart_avg_trade_size(summary):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = summary["avg_trade_size_usd"]
    colors = [SENTIMENT_COLORS[i] for i in vals.index]
    ax.bar(vals.index, vals.values, color=colors)
    ax.set_title("Average Trade Size (USD) by Market Sentiment")
    ax.set_ylabel("Avg Trade Size (USD)")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "03_avg_trade_size.png"), dpi=150)
    plt.close(fig)


def chart_daily_pnl_timeline(daily):
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(daily["date"], daily["daily_pnl"], color="#1f77b4", linewidth=1)
    ax1.set_ylabel("Daily Closed PnL (USD)", color="#1f77b4")
    ax1.axhline(0, color="grey", linewidth=0.6)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(daily["date"], daily["sentiment_value"], color="#e4572e", linewidth=1, alpha=0.6)
    ax2.set_ylabel("Fear/Greed Index (0-100)", color="#e4572e")
    ax2.tick_params(axis="y", labelcolor="#e4572e")

    ax1.set_title("Daily Trader PnL vs Fear/Greed Index Over Time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "04_daily_pnl_vs_sentiment_timeline.png"), dpi=150)
    plt.close(fig)


def chart_long_short_bias(bias_tab):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bias_tab = bias_tab.reindex(SENTIMENT_ORDER)
    bottom = np.zeros(len(bias_tab))
    for side, color in [("BUY", "#4C9A2A"), ("SELL", "#8B0000")]:
        if side in bias_tab.columns:
            ax.bar(bias_tab.index, bias_tab[side], bottom=bottom, label=side, color=color)
            bottom += bias_tab[side].values
    ax.set_title("Buy vs Sell Trade Mix by Sentiment")
    ax.set_ylabel("% of Trades")
    ax.legend()
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "05_long_short_bias.png"), dpi=150)
    plt.close(fig)


def chart_symbol_heatmap(pivot):
    cols = [c for c in SENTIMENT_ORDER if c in pivot.columns]
    data = pivot[cols]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(data.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=20)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_title("Top-10 Coins: Closed PnL (USD) by Sentiment")
    fig.colorbar(im, ax=ax, label="Closed PnL (USD)")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "06_symbol_sentiment_heatmap.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main():
    # Looks for data in ./data/ (repo layout) first, falls back to the
    # Claude.ai uploads path used during development.
    local_data = os.path.join(BASE, "data")
    if os.path.exists(os.path.join(local_data, "historical_data.csv")):
        uploads = local_data
    else:
        uploads = "/mnt/user-data/uploads"
    sentiment = load_sentiment(os.path.join(uploads, "fear_greed_index.csv"))
    trades = load_trades(os.path.join(uploads, "historical_data.csv"))

    merged = merge_datasets(trades, sentiment)
    merged = add_trade_flags(merged)

    print(f"Trades loaded: {len(trades):,}")
    print(f"Trades after merging with sentiment (date overlap): {len(merged):,}")
    print(f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")

    summary = sentiment_summary(merged)
    summary.to_csv(os.path.join(RESULTS_DIR, "sentiment_summary.csv"))
    print("\n=== Sentiment Summary ===")
    print(summary)

    daily = daily_pnl_vs_sentiment(merged)
    daily.to_csv(os.path.join(RESULTS_DIR, "daily_pnl_vs_sentiment.csv"), index=False)

    bias = long_short_bias(merged)
    bias.to_csv(os.path.join(RESULTS_DIR, "long_short_bias.csv"))
    print("\n=== Buy/Sell Bias (%) ===")
    print(bias)

    sym = symbol_breakdown(merged)
    sym.to_csv(os.path.join(RESULTS_DIR, "symbol_breakdown.csv"))

    leaders = trader_leaderboard(merged)
    leaders.to_csv(os.path.join(RESULTS_DIR, "trader_leaderboard.csv"))

    corr = sentiment_correlation(daily)
    print(f"\nCorrelation (sentiment value vs daily PnL): {corr:.4f}")

    contrarian = contrarian_check(summary)
    print("\n=== Fear vs Greed Aggregate ===")
    for k, v in contrarian.items():
        print(f"{k}: {v}")

    # Save headline numbers for report generation
    headline = pd.Series({
        "correlation_sentiment_vs_daily_pnl": round(corr, 4),
        **contrarian,
        "best_sentiment_by_win_rate": summary["win_rate_pct"].idxmax(),
        "worst_sentiment_by_win_rate": summary["win_rate_pct"].idxmin(),
        "best_sentiment_by_total_pnl": summary["total_closed_pnl"].idxmax(),
        "worst_sentiment_by_total_pnl": summary["total_closed_pnl"].idxmin(),
        "total_trades_analyzed": len(merged),
        "unique_traders": merged["Account"].nunique(),
        "date_range_start": str(merged["date"].min().date()),
        "date_range_end": str(merged["date"].max().date()),
    })
    headline.to_csv(os.path.join(RESULTS_DIR, "headline_stats.csv"))

    # Charts
    chart_total_pnl_by_sentiment(summary)
    chart_winrate_by_sentiment(summary)
    chart_avg_trade_size(summary)
    chart_daily_pnl_timeline(daily)
    chart_long_short_bias(bias)
    chart_symbol_heatmap(sym)

    print("\nAll results saved to ./results/, all charts saved to ./charts/")


if __name__ == "__main__":
    main()
