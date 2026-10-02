"""
backtest_ict.py -- backtests the EXACT live logic in scanner.py (v2).
Put it in the same folder as scanner.py.

Run:  python backtest_ict.py
Needs: pip install yfinance pandas numpy requests

Reports with-trend vs counter-trend, both halves of the data, per pair.
Outcomes: SL = -1R, TP = +3R, minus estimated spread/slippage cost.
Setups that never hit SL or TP by the end of data are excluded (counted).
"""

import numpy as np
import pandas as pd

import scanner

scanner.LOOKBACK = "60d"        # yfinance's max for 5m data
scanner.TREND_FILTER = False    # keep counter-trend trades so we can compare


def run():
    rows = []
    for ticker, name in scanner.PAIRS.items():
        try:
            trades = scanner.scan_pair(ticker, name)
        except Exception as e:
            print(f"[{name}] error: {e}")
            continue
        for t in trades:
            if t["score"] < scanner.MIN_ALERT_SCORE:
                continue
            if not t["in_session"] and name not in scanner.SESSION_EXEMPT_PAIRS:
                continue
            cost_pct = scanner.PAIR_COST_OVERRIDES.get(name, scanner.ROUND_TRIP_COST_PCT)
            cost_r = cost_pct * t["entry_price"] / t["risk"]
            if t["resolved"] == "TP":
                r = scanner.TARGET_R - cost_r
            elif t["resolved"] == "SL":
                r = -1.0 - cost_r
            else:
                r = np.nan
            rows.append({"pair": name, "time": t["entry_time"],
                         "with_trend": t["htf_aligned"], "r": r,
                         "open": t["resolved"] is None})
        print(f"[{name}] done")
    return pd.DataFrame(rows)


def summarize(df, label):
    d = df.dropna(subset=["r"])
    if d.empty:
        print(f"{label:<14} no closed trades")
        return
    wins = (d["r"] > 0).mean() * 100
    print(f"{label:<14} n={len(d):<4} win={wins:5.1f}%  avgR={d['r'].mean():+.3f}  totalR={d['r'].sum():+.1f}")


def main():
    df = run()
    if df.empty:
        print("No trades found.")
        return
    df = df.sort_values("time").reset_index(drop=True)
    print(f"\nUnresolved (excluded): {int(df['open'].sum())} of {len(df)}")
    mid = len(df) // 2

    for title, sub in (("ALL", df),
                       ("WITH-TREND", df[df.with_trend]),
                       ("COUNTER-TREND", df[~df.with_trend])):
        print(f"\n=== {title} ===")
        summarize(sub, "overall")
        if len(sub) >= 4:
            h = len(sub) // 2
            summarize(sub.iloc[:h], "first half")
            summarize(sub.iloc[h:], "second half")

    print("\n=== WITH-TREND by pair ===")
    wt = df[df.with_trend]
    for pair, g in wt.groupby("pair"):
        summarize(g, pair)

    print("\nVerdict guide: trade with-trend only if avgR > +0.10 after costs, "
          "n >= 50, and BOTH halves positive.")


if __name__ == "__main__":
    main()
