"""Plain 09:20 short straddle on data/upstox_7delta.db (NIFTY + SENSEX).

  * Every minute 09:20 .. --entry-last (default 09:20 only): sell ATM CE + PE (ATM by
    synthetic future), nearest expiry (the expiring contract on expiry day). Each entry
    is an independent straddle, 1 lot per leg.
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
from multiprocessing import Pool

import numpy as np

import straddle_920 as st


def run_day(job):
    db_path, index, day, expiry, entry_last, max_re, sl_fill = job
    st.SL_FILL = sl_fill
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    book = st.DayBook(db, index, expiry, day)
    if not len(book.strikes) or np.isnan(book.open[:, :, st.mi("09:20")]).all():
        return index, day, None, []
    cas = not np.isnan(book.high[:, :, st.mi("15:31"):]).all()
    m_exit = st.mi("15:35") if (cas and expiry != day) else st.mi("15:15")
    rows = []
    for m in range(st.mi("09:20"), st.mi(entry_last) + 1):
        for t in st.simulate_entry(book, index, m, 0.0, None, m_exit, max_re):
            rows.append(dict(attempt=t["attempt"], gross=t["gross"], charges=t["charges"],
                             net=t["net"], slot=st.hhmm(m), day=day, expiry=expiry,
                             expiry_day=expiry == day))
    return index, day, sum(r["net"] for r in rows), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/upstox_7delta.db")
    ap.add_argument("--max-re", type=int, default=1)
    ap.add_argument("--sl-fill", default="sl")
    ap.add_argument("--entry-last", default="09:20")
    a = ap.parse_args()
    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    jobs = []
    for index in ("NIFTY", "SENSEX"):
        exps = [r[0] for r in db.execute("SELECT DISTINCT expiry FROM contracts WHERE underlying=?"
                                         " AND n_candles > 0 ORDER BY 1", (index,))]
        days = [r[0] for r in db.execute(
            "SELECT DISTINCT date(o.ts,'unixepoch','+330 minutes') FROM option_candles o JOIN"
            " contracts c ON c.id=o.contract_id WHERE c.underlying=? ORDER BY 1", (index,))]
        for day in days:
            up = [e for e in exps if e >= day]
            if up:
                jobs.append((a.db, index, day, up[0], a.entry_last, a.max_re, a.sl_fill))
    result = {"NIFTY": ({}, []), "SENSEX": ({}, [])}
    with Pool(4) as pool:
        for index, day, net, rows in pool.imap_unordered(run_day, jobs, chunksize=4):
            if net is not None:
                result[index][0][day] = net
                result[index][1].extend(rows)
    for index in result:
        result[index] = (dict(sorted(result[index][0].items())), result[index][1])
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
        slot = collections.defaultdict(float)
        for t in rows:
            slot[t["slot"][:4] + "0"] += t["net"]
        if len(slot) > 1:
            print("  entry time:", ", ".join(f"{k} {x:,.0f}" for k, x in sorted(slot.items())))


if __name__ == "__main__":
    summarise(main())
