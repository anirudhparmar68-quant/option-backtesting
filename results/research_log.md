# Strategy research log (2-year Upstox data, Sep 2024 - Oct 2026)

All tests: next-minute execution at real option prices, statutory charges, 1 tick slippage
per leg per fill. PRE = Sep 2024 - Jun 2026, HOLD = Jul - Oct 2026 (holdout).

| # | Idea | Result | Why it failed / note |
|---|------|--------|----------------------|
| 1 | Gamma-neutral calendar (short near, long next straddle) | loss | Gamma-matched theta is ~equal (theta = -1/2 G s^2 F^2); intraday IV decline hurts the long-vega leg |
| 2 | Expiry-day OI pinning (fly at max-OI strike) | no edge | Close was nearer ATM than max-OI strike; moved toward it ~50% of the time |
| 3 | No-stop hedged fly, time exit (variance risk premium) | loss / ~0 | Intraday premium edge does not survive costs without stop management |
| 4 | NIFTY-SENSEX synthetic-futures pair trade | ~0 | Apparent 1-min mean reversion (lag-1 autocorr -0.28) is mostly stale-LTP noise; gone at next-minute fills |
| 5 | Opening-range breakout, long options | loss PRE (Sharpe -0.7..-1.6) | Buyers pay the variance premium; index breakouts mostly revert |
| 6 | Opening-range breakout, synthetic futures | loss PRE | Breakouts revert in PRE; pattern flips in HOLD (not robust) |
| 7 | Overnight short straddle (15:15 -> 09:20) | NIFTY: weak + (PRE Sharpe 0.8, HOLD 2.5); SENSEX: looked like Sharpe 5.7 but INVALID | SENSEX data holds only ATM+-3 strikes: 35 big-gap nights (avg gap 1.55%) silently dropped = survivorship bias |
| - | Close-confirmed staggered straddle (CCS) | robust: ret/DD 9.2, Sharpe 2.4 | Built on the user's straddle; edge real once stops use closes + next-minute fills |
