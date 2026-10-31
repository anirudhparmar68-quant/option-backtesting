"""Plain 09:20 short straddle on data/upstox_7delta.db (NIFTY + SENSEX).

  * 09:20: sell ATM CE + PE (ATM by synthetic future), nearest expiry (the expiring
    contract on expiry day). 1 lot each.
  * SL 20% per leg; after one leg's SL the other leg's SL moves to cost.
  * Both legs stopped -> re-entry at once (next minute, new ATM), not after 14:50.
    MAX_RE re-entries (default 1; --max-re 99 = unlimited).
  * Exit: 15:15 before the closing auction (CAS); after CAS 15:15 on expiry day and
    15:35 on other days (CAS detected per day: bars after 15:30).
  * No slippage; statutory charges only. SL hit when the minute high >= SL, filled at the
    SL price (or the open if the minute opens above it) - see straddle_920.py.

Usage: python backtests/straddle_plain.py [--max-re 1] [--sl-fill sl|close|high]
"""

import argparse
import collections
import sqlite3

import numpy as np

import straddle_920 as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/upstox_7delta.db")
    ap.add_argument("--max-re", type=int, default=1)
    ap.add_argument("--sl-fill", default="sl")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    st.SL_FILL = a.sl_fill
    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    result = {}
    for index in ("NIFTY", "SENSEX"):
        exps = [r[0] for r in db.execute("SELECT DISTINCT expiry FROM contracts WHERE underlying=?"
                                         " AND n_candles > 0 ORDER BY 1", (index,))]
        days = [r[0] for r in db.execute(
            "SELECT DISTINCT date(o.ts,'unixepoch','+330 minutes') FROM option_candles o JOIN"
            " contracts c ON c.id=o.contract_id WHERE c.underlying=? ORDER BY 1", (index,))]
        daily, rows = {}, []
        for day in days:
            up = [e for e in exps if e >= day]
            if not up:
                continue
            book = st.DayBook(db, index, up[0], day)
            if not len(book.strikes) or np.isnan(book.open[:, :, st.mi("09:20")]).all():
                continue
            cas = not np.isnan(book.high[:, :, st.mi("15:31"):]).all()
            m_exit = st.mi("15:35") if (cas and up[0] != day) else st.mi("15:15")
            trades = st.simulate_entry(book, index, st.mi("09:20"), 0.0, None, m_exit, a.max_re)
            daily[day] = sum(t["net"] for t in trades)
            rows += [dict(t, day=day, expiry=up[0], expiry_day=up[0] == day) for t in trades]
        result[index] = (daily, rows)
    return result


def summarise(result):
    for index, (daily, rows) in result.items():
        v = np.array(list(daily.values()))
        eq = np.cumsum(v)
        dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq))
        re = sum(1 for t in rows if t["attempt"] > 0) // 2
        gross, chg = sum(t["gross"] for t in rows), sum(t["charges"] for t in rows)
        exp_days = {t["day"] for t in rows if t["expiry_day"]}
        ve = sum(daily[d] for d in exp_days)
        print(f"{index}: {len(v)} days, re-entries {re}, gross {gross:,.0f}, charges {chg:,.0f},"
              f" NET {v.sum():,.0f}, maxDD {dd:,.0f}, win days {(v > 0).mean():.0%},"
              f" expiry-day net {ve:,.0f} ({len(exp_days)} days)")
        mon = collections.defaultdict(float)
        for d, x in daily.items():
            mon[d[:7]] += x
        print("  months:", ", ".join(f"{m} {x:,.0f}" for m, x in sorted(mon.items())))


if __name__ == "__main__":
    summarise(main())
