"""
scanner.py -- live ICT-style scanner (liquidity sweep -> displacement/BOS ->
FVG -> retracement) matching the backtested combo:
  score 9-10, in-session (07-20 UTC), 3R target, 12-pair shortlist.

This mirrors the detection logic from ict_backtest.py but runs on a short
rolling window of recent data (not 60 days) since live use only needs
enough history for HTF trend / PDH-PDL / recent swings, not a full backtest.

Alerting is stateless: on each run, only setups whose ENTRY occurred within
the last ALERT_WINDOW_MINUTES get sent to Telegram. As long as this script
runs at least every ALERT_WINDOW_MINUTES, each real entry gets alerted
exactly once without needing to persist state between runs.

Env vars required: TELEGRAM_TOKEN, CHAT_ID
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ----------------------------------------------------------------------
# CONFIG -- locked to the backtested combo. Change with care and re-test.
# ----------------------------------------------------------------------

PAIRS = {
    "GC=F": "XAUUSD", "SI=F": "XAGUSD",
    "^GDAXI": "GER40", "^N225": "JP225",
    "^NDX": "US100", "^DJI": "US30", "^GSPC": "US500",
    "DOGE-USD": "DOGEUSDT", "LTC-USD": "LTCUSDT",
    "AUDJPY=X": "AUDJPY", "USDJPY=X": "USDJPY", "EURUSD=X": "EURUSD",
}
# JP225 trades mostly outside 07-20 UTC -- kept as an explicit exception to
# the in-session filter rather than silently losing it. Remove this set to
# go back to strict in-session-only for every pair.
SESSION_EXEMPT_PAIRS = {"JP225"}

LOOKBACK = "5d"           # only needs recent history, not the full backtest window
ENTRY_INTERVAL = "5m"

FRACTAL_N = 2
SWING_ACTIVE_BARS = 300
DISPLACEMENT_LOOKAHEAD = 6
DISPLACEMENT_BODY_MULT = 1.5
SHORT_STRUCT_LOOKBACK = 20
FVG_SEARCH_WINDOW = 4
FVG_MIN_ATR_RATIO = 0.10
RETRACE_LOOKAHEAD = 40
ATR_PERIOD = 14

TARGET_R = 3
MIN_ALERT_SCORE = 9
SESSION_START_UTC = 7
SESSION_END_UTC = 20

ROUND_TRIP_COST_PCT = 0.0003
PAIR_COST_OVERRIDES = {
    "EURUSD": 0.00015, "USDJPY": 0.00015, "AUDJPY": 0.00025,
    "XAUUSD": 0.00015, "XAGUSD": 0.00050,
    "US30": 0.00010, "US100": 0.00010, "US500": 0.00010,
    "GER40": 0.00015, "JP225": 0.00015,
    "DOGEUSDT": 0.00120, "LTCUSDT": 0.00080,
}
MIN_RISK_VS_COST_MULTIPLE = 3

ALERT_WINDOW_MINUTES = 20  # only alert on entries newer than this


# ----------------------------------------------------------------------
# DATA HELPERS (same as ict_backtest.py)
# ----------------------------------------------------------------------

def fetch(ticker):
    df = yf.download(ticker, period=LOOKBACK, interval=ENTRY_INTERVAL,
                      progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def add_htf_trend(df):
    h1 = df["Close"].resample("1h").last().dropna()
    ema50 = h1.ewm(span=50, min_periods=10).mean()
    trend = pd.Series(np.where(h1 > ema50, "up", "down"), index=h1.index)
    trend = trend.reindex(df.index, method="ffill")
    df = df.copy()
    df["htf_trend"] = trend
    return df


def add_daily_weekly_levels(df):
    daily = df.resample("1D").agg({"High": "max", "Low": "min"}).dropna()
    daily["PDH"] = daily["High"].shift(1)
    daily["PDL"] = daily["Low"].shift(1)
    weekly = df.resample("1W").agg({"High": "max", "Low": "min"}).dropna()
    weekly["PWH"] = weekly["High"].shift(1)
    weekly["PWL"] = weekly["Low"].shift(1)
    df = df.copy()
    df["PDH"] = daily["PDH"].reindex(df.index, method="ffill")
    df["PDL"] = daily["PDL"].reindex(df.index, method="ffill")
    df["PWH"] = weekly["PWH"].reindex(df.index, method="ffill")
    df["PWL"] = weekly["PWL"].reindex(df.index, method="ffill")
    return df


def add_fractals(df, n=FRACTAL_N):
    win = 2 * n + 1
    df = df.copy()
    df["swing_high"] = df["High"] == df["High"].rolling(win, center=True).max()
    df["swing_low"] = df["Low"] == df["Low"].rolling(win, center=True).min()
    return df


def add_atr(df, period=ATR_PERIOD):
    df = df.copy()
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    return df


class Level:
    __slots__ = ("idx", "price", "kind", "major", "swept")
    def __init__(self, idx, price, kind, major):
        self.idx, self.price, self.kind, self.major = idx, price, kind, major
        self.swept = False


def find_bos_reference(highs, lows, i, direction, lookback):
    lo = max(0, i - lookback)
    seg = highs[lo:i] if direction == "bull" else lows[lo:i]
    return (seg.max() if direction == "bull" else seg.min()) if len(seg) else None


def find_fvg(df, disp_idx, direction, atr):
    lo = max(1, disp_idx - FVG_SEARCH_WINDOW)
    hi = min(len(df) - 2, disp_idx + FVG_SEARCH_WINDOW)
    highs, lows = df["High"].values, df["Low"].values
    for k in range(lo, hi):
        if direction == "bull":
            gap_low, gap_high = highs[k - 1], lows[k + 1]
        else:
            gap_high, gap_low = lows[k - 1], highs[k + 1]
        if gap_high > gap_low:
            size = gap_high - gap_low
            clean = atr[k] and size >= FVG_MIN_ATR_RATIO * atr[k]
            return gap_low, gap_high, clean, k + 1
    return None


def opposing_target(levels, entry_price, direction, min_reward):
    best = None
    for lv in levels:
        if lv.swept:
            continue
        if direction == "bull" and lv.kind == "high" and lv.price > entry_price:
            reward = lv.price - entry_price
        elif direction == "bear" and lv.kind == "low" and lv.price < entry_price:
            reward = entry_price - lv.price
        else:
            continue
        if reward >= min_reward and (best is None or reward < best):
            best = reward
    return best


def try_build_trade(df, sweep_idx, direction, swept_level, active_levels,
                     highs, lows, closes, atr, avg_body, htf_trend, idx, name):
    n = len(df)

    inducement = False
    for k in range(sweep_idx + 1, min(sweep_idx + 4, n - 1)):
        if direction == "bull" and closes[k] < closes[sweep_idx]:
            inducement = True; break
        if direction == "bear" and closes[k] > closes[sweep_idx]:
            inducement = True; break

    bos_ref = find_bos_reference(highs, lows, sweep_idx, direction, SHORT_STRUCT_LOOKBACK)
    if bos_ref is None:
        return None

    disp_idx = None
    for k in range(sweep_idx + 1, min(sweep_idx + DISPLACEMENT_LOOKAHEAD, n - 1)):
        body = abs(closes[k] - df["Open"].values[k])
        ab = avg_body[k]
        if not ab or np.isnan(ab):
            continue
        big_body = body >= DISPLACEMENT_BODY_MULT * ab
        if direction == "bull" and big_body and closes[k] > bos_ref:
            disp_idx = k; break
        if direction == "bear" and big_body and closes[k] < bos_ref:
            disp_idx = k; break
    if disp_idx is None:
        return None

    fvg = find_fvg(df, disp_idx, direction, atr)
    if fvg is None:
        return None
    fvg_low, fvg_high, fvg_clean, fvg_confirm_idx = fvg
    if fvg_high <= fvg_low:
        return None

    entry_idx = None
    for k in range(fvg_confirm_idx + 1, min(fvg_confirm_idx + RETRACE_LOOKAHEAD, n - 1)):
        if lows[k] <= fvg_high and highs[k] >= fvg_low:
            entry_idx = k; break
    if entry_idx is None:
        return None

    entry_price = (fvg_low + fvg_high) / 2.0
    buffer = 0.1 * (atr[sweep_idx] if atr[sweep_idx] and not np.isnan(atr[sweep_idx]) else 0)
    if direction == "bull":
        sl = min(lows[sweep_idx], fvg_low) - buffer
        risk = entry_price - sl
    else:
        sl = max(highs[sweep_idx], fvg_high) + buffer
        risk = sl - entry_price
    if risk <= 0:
        return None

    cost_pct = PAIR_COST_OVERRIDES.get(name, ROUND_TRIP_COST_PCT)
    if risk < MIN_RISK_VS_COST_MULTIPLE * cost_pct * entry_price:
        return None

    score = 4  # sweep + displacement + BOS + retracement (mandatory)
    trend_at_entry = htf_trend[entry_idx]
    htf_aligned = (direction == "bull" and trend_at_entry == "up") or \
                  (direction == "bear" and trend_at_entry == "down")
    score += int(htf_aligned)
    score += int(swept_level.major)
    score += int(inducement)
    score += int(bool(fvg_clean))
    target_dist = opposing_target(active_levels, entry_price, direction, 1.2 * risk)
    score += int(target_dist is not None)
    entry_hour = idx[entry_idx].hour
    in_session = SESSION_START_UTC <= entry_hour <= SESSION_END_UTC
    score += int(in_session)

    tp = entry_price + TARGET_R * risk if direction == "bull" else entry_price - TARGET_R * risk

    return {
        "pair": name, "direction": "BUY" if direction == "bull" else "SELL",
        "entry_time": idx[entry_idx], "entry_price": entry_price,
        "sl": sl, "tp": tp, "risk": risk, "score": score,
        "in_session": in_session,
    }


def scan_pair(ticker, name):
    raw = fetch(ticker)
    if raw.empty or len(raw) < 300:
        return []

    df = add_htf_trend(raw)
    df = add_daily_weekly_levels(df)
    df = add_fractals(df)
    df = add_atr(df)

    highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values
    atr = df["ATR"].values
    swing_high, swing_low = df["swing_high"].values, df["swing_low"].values
    htf_trend = df["htf_trend"].values
    pdh, pdl, pwh, pwl = df["PDH"].values, df["PDL"].values, df["PWH"].values, df["PWL"].values
    idx = df.index
    avg_body = pd.Series(np.abs(closes - df["Open"].values)).rolling(20).mean().values

    active_levels = []
    n = FRACTAL_N
    trades = []

    for i in range(n, len(df) - 1):
        confirm_idx = i - n
        if confirm_idx >= 0:
            if swing_high[confirm_idx]:
                active_levels.append(Level(confirm_idx, highs[confirm_idx], "high", False))
            if swing_low[confirm_idx]:
                active_levels.append(Level(confirm_idx, lows[confirm_idx], "low", False))
        for price, kind in ((pdh[i], "high"), (pdl[i], "low"),
                             (pwh[i], "high"), (pwl[i], "low")):
            if not np.isnan(price):
                already = any(lv.major and lv.price == price and lv.kind == kind
                              for lv in active_levels[-8:])
                if not already:
                    active_levels.append(Level(i, price, kind, True))
        active_levels = [lv for lv in active_levels
                          if lv.swept is False and i - lv.idx < SWING_ACTIVE_BARS] \
                          + [lv for lv in active_levels if lv.swept]

        bar_low, bar_high, bar_close = lows[i], highs[i], closes[i]

        for lv in active_levels:
            if lv.swept or lv.kind != "low" or lv.idx >= i:
                continue
            if bar_low < lv.price and bar_close > lv.price:
                lv.swept = True
                trade = try_build_trade(df, i, "bull", lv, active_levels, highs, lows,
                                         closes, atr, avg_body, htf_trend, idx, name)
                if trade:
                    trades.append(trade)
                break

        for lv in active_levels:
            if lv.swept or lv.kind != "high" or lv.idx >= i:
                continue
            if bar_high > lv.price and bar_close < lv.price:
                lv.swept = True
                trade = try_build_trade(df, i, "bear", lv, active_levels, highs, lows,
                                         closes, atr, avg_body, htf_trend, idx, name)
                if trade:
                    trades.append(trade)
                break

    return trades


# ----------------------------------------------------------------------
# TELEGRAM
# ----------------------------------------------------------------------

def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={"chat_id": chat_id, "text": text,
                                     "parse_mode": "Markdown"}, timeout=15)
    if resp.status_code != 200:
        print(f"Telegram send failed: {resp.status_code} {resp.text}")


def format_alert(trade):
    emoji = "🟢" if trade["direction"] == "BUY" else "🔴"
    session_tag = "In-session" if trade["in_session"] else "Off-session (JP225 exception)"
    return (
        f"{emoji} *ICT SETUP -- {trade['pair']}*\n"
        f"Direction: {trade['direction']}\n"
        f"Score: {trade['score']}/10\n"
        f"Session: {session_tag}\n"
        f"Entry: {trade['entry_price']:.5f}\n"
        f"SL: {trade['sl']:.5f}\n"
        f"TP (3R): {trade['tp']:.5f}\n"
        f"Risk distance: {trade['risk']:.5f}\n"
        f"Entry time: {trade['entry_time'].strftime('%Y-%m-%d %H:%M UTC')}"
    )


def main():
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("CHAT_ID")
    if not token or not chat_id:
        print("Missing TELEGRAM_TOKEN or CHAT_ID env vars.")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=ALERT_WINDOW_MINUTES)

    alerts_sent = 0
    for ticker, name in PAIRS.items():
        try:
            trades = scan_pair(ticker, name)
        except Exception as e:
            print(f"[{name}] scan error: {e}")
            continue

        for t in trades:
            if t["score"] < MIN_ALERT_SCORE:
                continue
            if not t["in_session"] and name not in SESSION_EXEMPT_PAIRS:
                continue
            if t["entry_time"] < cutoff:
                continue  # not a fresh entry -- already would have been alerted
            send_telegram(token, chat_id, format_alert(t))
            alerts_sent += 1
            print(f"Alert sent: {name} {t['direction']} score={t['score']} "
                  f"entry_time={t['entry_time']}")

    print(f"Run complete. {alerts_sent} alert(s) sent at {now.isoformat()}.")


if __name__ == "__main__":
    main()
