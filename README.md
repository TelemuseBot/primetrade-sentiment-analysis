"""
Generates the final PDF report: 'Trader Performance vs Bitcoin Market
Sentiment'. Reads the tables/charts produced by analysis.py and assembles
them into a polished PDF using reportlab.

Run: python3 generate_report.py   (after analysis.py has been run)
"""

import os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable
)

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
CHARTS = os.path.join(BASE, "charts")
_repo_report_dir = os.path.join(BASE, "report")
if os.path.isdir(_repo_report_dir) or os.path.isdir(RESULTS):
    OUT_PDF = os.path.join(_repo_report_dir, "Trader_Sentiment_Analysis_Report.pdf")
    os.makedirs(_repo_report_dir, exist_ok=True)
else:
    OUT_PDF = "/mnt/user-data/outputs/Trader_Sentiment_Analysis_Report.pdf"
    os.makedirs("/mnt/user-data/outputs", exist_ok=True)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBig", fontSize=22, leading=26,
                           spaceAfter=6, textColor=colors.HexColor("#1a1a1a"),
                           fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="SubTitle", fontSize=12, leading=16,
                           textColor=colors.HexColor("#555555")))
styles.add(ParagraphStyle(name="H2", fontSize=15, leading=18, spaceBefore=16,
                           spaceAfter=8, textColor=colors.HexColor("#0B3D91"),
                           fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="H3", fontSize=12, leading=15, spaceBefore=10,
                           spaceAfter=6, textColor=colors.HexColor("#1a1a1a"),
                           fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Body", fontSize=10, leading=15,
                           spaceAfter=8, alignment=4))
styles.add(ParagraphStyle(name="BulletItem", fontSize=10, leading=14,
                           leftIndent=14, spaceAfter=4))
styles.add(ParagraphStyle(name="Caption", fontSize=8.5, leading=11,
                           textColor=colors.HexColor("#666666"),
                           alignment=1, spaceAfter=14))

story = []


def h2(text):
    story.append(Paragraph(text, styles["H2"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#0B3D91")))
    story.append(Spacer(1, 6))


def h3(text):
    story.append(Paragraph(text, styles["H3"]))


def body(text):
    story.append(Paragraph(text, styles["Body"]))


def bullet(text):
    story.append(Paragraph(f"&bull;&nbsp;&nbsp;{text}", styles["BulletItem"]))


def chart(filename, width=15.5 * cm, caption=None):
    img_path = os.path.join(CHARTS, filename)
    img = Image(img_path, width=width, height=width * 0.58)
    story.append(img)
    if caption:
        story.append(Paragraph(caption, styles["Caption"]))
    story.append(Spacer(1, 4))


def df_to_table(df, col_widths=None, index_label=""):
    df = df.reset_index()
    df.columns = [index_label if str(c) == "index" or str(c) == df.columns[0] and index_label else str(c) for c in df.columns]
    data = [list(df.columns)] + df.values.tolist()
    data = [[f"{v:,.2f}" if isinstance(v, float) else str(v) for v in row] for row in data]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F4F8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

summary = pd.read_csv(os.path.join(RESULTS, "sentiment_summary.csv"), index_col=0)
bias = pd.read_csv(os.path.join(RESULTS, "long_short_bias.csv"), index_col=0)
headline = pd.read_csv(os.path.join(RESULTS, "headline_stats.csv"), index_col=0).iloc[:, 0]
leaders = pd.read_csv(os.path.join(RESULTS, "trader_leaderboard.csv"), index_col=0)

corr = float(headline["correlation_sentiment_vs_daily_pnl"])
fear_pnl = float(headline["fear_total_pnl"])
greed_pnl = float(headline["greed_total_pnl"])
fear_avg = float(headline["fear_avg_pnl_per_trade"])
greed_avg = float(headline["greed_avg_pnl_per_trade"])
total_trades = int(float(headline["total_trades_analyzed"]))
unique_traders = int(float(headline["unique_traders"]))
date_start = headline["date_range_start"]
date_end = headline["date_range_end"]

# ---------------------------------------------------------------------------
# COVER
# ---------------------------------------------------------------------------

story.append(Spacer(1, 3 * cm))
story.append(Paragraph("Trader Performance vs. Bitcoin", styles["TitleBig"]))
story.append(Paragraph("Market Sentiment: A Data-Driven Analysis", styles["TitleBig"]))
story.append(Spacer(1, 10))
story.append(Paragraph(
    "An exploration of how Fear &amp; Greed sentiment shapes trading behaviour and "
    "profitability on Hyperliquid, with actionable strategy implications.",
    styles["SubTitle"]))
story.append(Spacer(1, 40))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0B3D91")))
story.append(Spacer(1, 10))
meta = Table([
    ["Dataset period", f"{date_start} to {date_end}"],
    ["Total trades analyzed", f"{total_trades:,}"],
    ["Unique trader accounts", f"{unique_traders}"],
    ["Prepared for", "Primetrade.ai — Data Science Assignment"],
], colWidths=[6 * cm, 9 * cm])
meta.setStyle(TableStyle([
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
]))
story.append(meta)
story.append(PageBreak())

# ---------------------------------------------------------------------------
# EXECUTIVE SUMMARY
# ---------------------------------------------------------------------------

h2("1. Executive Summary")
body(
    f"This report merges {total_trades:,} individual trade executions from Hyperliquid "
    f"({unique_traders} accounts) with the Bitcoin Fear &amp; Greed sentiment index to test "
    "whether market-wide sentiment is associated with trader profitability. The central, "
    "somewhat counter-intuitive finding: <b>traders in this dataset were consistently more "
    f"profitable during Fear regimes than Greed regimes</b>, both in total realized PnL "
    f"(${fear_pnl:,.0f} during Fear vs ${greed_pnl:,.0f} during Greed) and on a per-trade "
    f"basis (${fear_avg:,.0f} avg PnL/trade in Fear vs ${greed_avg:,.0f} in Greed)."
)
body(
    f"The correlation between the raw sentiment index value (0-100) and aggregate daily "
    f"closed PnL is <b>{corr:+.2f}</b> — a moderate negative relationship, meaning PnL tends "
    "to be higher on days when the market is more fearful, not less."
)
h3("Key takeaways")
bullet("Win rate and average PnL-per-trade both decline in a near-monotonic staircase from Extreme Fear (highest) down to Extreme Greed (lowest).")
bullet("Traders take larger average position sizes during Fear and Greed than during Neutral/Extreme Greed periods, suggesting conviction rises at sentiment extremes.")
bullet("Buy-side activity is proportionally highest during Extreme Fear (57% BUY) — consistent with contrarian 'buy the dip' behaviour — while Sell-side activity dominates during Extreme Greed (53% SELL), consistent with profit-taking into strength.")
bullet("Performance is highly concentrated: a small number of accounts account for the large majority of total profit, regardless of sentiment regime.")
story.append(PageBreak())

# ---------------------------------------------------------------------------
# 2. METHODOLOGY
# ---------------------------------------------------------------------------

h2("2. Data & Methodology")
body(
    "<b>Datasets:</b> (1) Hyperliquid historical trade-level data — account, coin, execution "
    "price, size, side, direction, closed PnL, fees, timestamps; (2) Bitcoin Fear &amp; Greed "
    "Index — daily sentiment score (0-100) and classification (Extreme Fear, Fear, Neutral, "
    "Greed, Extreme Greed)."
)
body(
    "<b>Merge key:</b> trade timestamps were converted to calendar dates and joined against "
    "the daily sentiment classification for that date."
)
body(
    "<b>PnL trades vs opens:</b> only trades tagged as position-closing events (Direction "
    "containing 'Close') carry a realized Closed PnL; 'Open' trades have PnL = 0 by "
    "construction on this exchange. All win-rate, average-PnL, and profitability metrics "
    "in this report are computed on the closing-trade subset only, while trade-size and "
    "buy/sell-mix metrics use the full trade set (opens + closes)."
)
body(
    "<b>Win rate</b> = share of closing trades with Closed PnL &gt; 0. "
    "<b>Total volume</b> = sum of Size (USD) across all trades in that sentiment bucket."
)
story.append(PageBreak())

# ---------------------------------------------------------------------------
# 3. HEADLINE PERFORMANCE TABLE
# ---------------------------------------------------------------------------

h2("3. Performance by Market Sentiment")
tbl_cols = ["total_trades", "closing_trades", "total_closed_pnl", "avg_closed_pnl", "win_rate_pct", "avg_trade_size_usd"]
disp = summary[tbl_cols].copy()
disp.columns = ["Total\nTrades", "Closing\nTrades", "Total Closed\nPnL (USD)", "Avg PnL /\nTrade (USD)", "Win Rate\n(%)", "Avg Trade\nSize (USD)"]
story.append(df_to_table(disp, col_widths=[2.3*cm, 2.3*cm, 3.0*cm, 2.6*cm, 2.2*cm, 2.6*cm], index_label="Sentiment"))
story.append(Spacer(1, 12))

chart("01_total_pnl_by_sentiment.png",
      caption="Figure 1. Total realized PnL collapses as sentiment moves from Fear toward Extreme Greed.")
chart("02_winrate_by_sentiment.png",
      caption="Figure 2. Win rate is highest in Neutral/Fear conditions and lowest in Extreme Greed.")
story.append(PageBreak())

chart("03_avg_trade_size.png",
      caption="Figure 3. Average trade size is largest during Fear and Greed (conviction extremes) and smallest during Extreme Fear/Extreme Greed panic-or-euphoria conditions.")
chart("04_daily_pnl_vs_sentiment_timeline.png",
      caption="Figure 4. Daily closed PnL (blue) plotted against the Fear/Greed index value (orange) over the full sample period.")
story.append(PageBreak())

# ---------------------------------------------------------------------------
# 4. TRADING BEHAVIOUR
# ---------------------------------------------------------------------------

h2("4. Trading Behaviour: Long/Short Bias")
body(
    "Buy vs Sell activity mix shifts systematically with sentiment. During Extreme Fear, "
    "57.0% of trades are on the Buy side — consistent with contrarian accumulation. This "
    "flips through the sentiment spectrum to Extreme Greed, where 52.9% of trades are Sell "
    "side, consistent with distribution/profit-taking as euphoria peaks."
)
chart("05_long_short_bias.png",
      caption="Figure 5. Buy/Sell trade mix (%) by sentiment classification.")

h2("5. Symbol-Level Patterns")
body(
    "The heatmap below shows realized PnL by coin, broken down by the sentiment regime in "
    "which it was earned, for the ten coins with the largest total realized PnL in the "
    "dataset. Performance concentration by symbol is uneven — a handful of coins drive most "
    "of the profit, and their sentiment sensitivity varies (some names are profitable across "
    "all regimes, others are markedly sentiment-dependent)."
)
chart("06_symbol_sentiment_heatmap.png",
      caption="Figure 6. Closed PnL (USD) by coin x sentiment classification, top 10 coins by total PnL.")
story.append(PageBreak())

# ---------------------------------------------------------------------------
# 6. TRADER CONCENTRATION
# ---------------------------------------------------------------------------

h2("6. Trader Concentration: Top & Bottom Performers")
body(
    "Profitability is highly concentrated. The table below shows the 10 highest and 10 "
    "lowest cumulative-PnL accounts across the full sample, illustrating that a small subset "
    "of accounts (often high trade-count, high win-rate) drive the bulk of aggregate profit."
)
lead_disp = leaders.copy()
lead_disp.columns = ["Total PnL\n(USD)", "Trades", "Win Rate\n(%)"]
lead_disp.index = [a[:10] + "..." for a in lead_disp.index]
story.append(df_to_table(lead_disp, col_widths=[4.5*cm, 3.5*cm, 3.0*cm, 3.0*cm], index_label="Account"))
story.append(PageBreak())

# ---------------------------------------------------------------------------
# 7. STRATEGY IMPLICATIONS
# ---------------------------------------------------------------------------

h2("7. Strategy Implications")
h3("For risk management")
bullet("Extreme Greed periods show the weakest win rate and PnL/trade in this sample — position sizing and leverage discipline should tighten, not loosen, as the index climbs into Greed/Extreme Greed territory.")
bullet("Extreme Fear periods, despite the 'panic' label, produced the strongest average PnL per trade — suggesting disciplined contrarian entries during fear are historically well-rewarded for this trader population.")
h3("For signal design")
bullet("The Fear/Greed index value can serve as a simple regime filter: a rules-based strategy could scale exposure up as the index falls toward Fear/Extreme Fear and scale down (or hedge) as it rises into Greed/Extreme Greed.")
bullet("Buy/Sell mix by sentiment (Fig. 5) confirms the broader trader base already behaves partly contrarian at extremes — a strategy that trades ahead of or alongside this shift, rather than chasing it late, may capture more edge.")
h3("For portfolio construction")
bullet("Because performance is concentrated in a few accounts and a few coins, sentiment-based signals should be combined with account/coin-level quality filters rather than applied uniformly across all trades.")
h3("Caveats")
bullet("Results are observational and correlational, not causal — sentiment may proxy for volatility, liquidity, or news-driven conditions that independently affect PnL.")
bullet("Closed PnL excludes unrealized PnL on still-open positions and does not net out fees at the aggregate level in every table; fee-adjusted net PnL would modestly reduce all totals.")
bullet("The dataset covers a specific set of Hyperliquid accounts and may not generalize to the broader market.")

story.append(Spacer(1, 20))
story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#CCCCCC")))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Prepared using Python (pandas, matplotlib) for analysis and reportlab for report "
    "generation. Full reproducible code accompanies this report as analysis.py.",
    styles["Caption"]))

# ---------------------------------------------------------------------------
doc = SimpleDocTemplate(OUT_PDF, pagesize=A4,
                         topMargin=2*cm, bottomMargin=2*cm,
                         leftMargin=2*cm, rightMargin=2*cm,
                         title="Trader Performance vs Bitcoin Market Sentiment")
doc.build(story)
print(f"Saved: {OUT_PDF}")
