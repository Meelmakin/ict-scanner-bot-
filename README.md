# ict-scanner-bot

Live Telegram scanner implementing an ICT-style checklist strategy:
**liquidity sweep → inducement → displacement/BOS → Fair Value Gap → retracement → entry.**

Runs on GitHub Actions every 15 minutes (`workflow_dispatch` also supported
for manual runs), same pattern as `smc-bot-azu` and `structure-engulf-bot`.

## ⚠️ Status: ALERT-ONLY -- not live-traded

This bot is in forward-validation. Backtest results below were promising,
but only cover a 60-day window. Track real alerts against real outcomes
for several weeks before risking capital on it, the same way break-retest
and structure-engulf were validated before going live.

## Config (locked to the backtested combo -- change with care)

| Setting | Value |
|---|---|
| Min alert score | 9 / 10 |
| Target | 3R |
| Session filter | 07:00–20:00 UTC (in-session only) |
| Session exception | JP225 alerts regardless of session (see below) |
| Pairs | XAUUSD, XAGUSD, GER40, JP225, US100, US30, US500, DOGEUSDT, LTCUSDT, AUDJPY, USDJPY, EURUSD |

### Why these settings
- **28 pairs tested, narrowed to these 12** -- the rest were flat or
  negative expectancy at score≥7 (EURAUD, NZDUSD, USDCHF, GBPJPY were
  clearly negative; several others had too few trades to judge).
- **Score ≥9 instead of ≥7** -- score 9-10 setups showed 0.389-0.488R
  expectancy vs 0.234-0.269R for the full score≥7 set. Narrower filter,
  meaningfully better per-trade quality.
- **JP225 kept as a session exception** -- it was a strong performer
  (0.84R expectancy at score≥7) but trades mostly outside 07-20 UTC.
  Excluding it under a strict session filter would have thrown away a
  genuinely good pair for the sake of a filter that doesn't fit its
  trading hours.
- **3R over 2R** -- higher expectancy in every backtest cut (full set,
  score buckets, and the final combo), at the cost of a lower win rate.

## Backtest results that justified this config

60-day window, 5-minute data, yfinance.

**Score ≥9, in-session, these 12 pairs, target 3R:**
- 330 trades, 45.2% win rate, profit factor 1.92
- Expectancy: **+0.567R per trade**
- Total: +187R, max drawdown -10.24R, max 7 consecutive losses

**Split-half consistency check** (first half of window vs second half):
- First half: 141 trades, +0.469R expectancy
- Second half: 189 trades, +0.639R expectancy
- Both positive, second half stronger -- not a result concentrated in one
  short burst.

**Caveats on the above, honestly stated:**
- Pair list and score threshold were both chosen by looking at
  performance *within this same 60-day window* -- the split check confirms
  temporal consistency, but doesn't fully rule out pair selection by
  hindsight. Real out-of-sample data (or forward alert-only tracking) is
  the next real test.
- A few pairs in the mix contribute most of the edge (XAUUSD, US100);
  a few contribute almost nothing (DOGEUSDT, AUDJPY, USDJPY). Kept in
  deliberately rather than re-trimmed, to avoid over-fitting the backtest
  further.
- No slippage/spread model beyond a rough per-pair estimate
  (`PAIR_COST_OVERRIDES` in `scanner.py`) -- replace with real broker
  numbers when available.

## How it works

`scanner.py` re-runs the full detection logic on a rolling 5-day window
each time it fires. Alerts are stateless: only setups whose entry
occurred within the last 20 minutes get sent, so each real signal fires
exactly once as long as the scan runs at least that often.

## Setup

1. Add repo secrets: `TELEGRAM_TOKEN`, `CHAT_ID`
2. Confirm the workflow runs clean via a manual `workflow_dispatch` trigger
   before trusting the schedule
3. (Optional, same as the other two bots) add an external cron-job.org
   trigger hitting this repo's `workflow_dispatch` endpoint every 15
   minutes, since GitHub's built-in scheduler can queue/delay under load
