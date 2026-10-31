"""Staggered short straddle backtest on Upstox 1-minute data (data/upstox_options.db).

Rules (per straddle, 1 lot):
  * Entries every minute from 09:20 to 11:00 -> 101 independent straddles a day.
  * Strike = ATM from synthetic future: at the strike where |CE - PE| is smallest,
    F = K + CE - PE; ATM = listed strike nearest F.
  * Sell ATM CE + PE. SL on each leg = entry premium * 1.20.
  * When one leg's SL hits, the other leg's SL moves to its entry price (cost),
    effective from the next minute.
  * Both legs stopped -> one re-entry at the next minute (new synthetic ATM),
    only if that minute is <= 14:50. Re-entry follows the same rules; no further
    re-entry after it.
  * Anything still open is bought back at 15:35.
  * Expiry used: nearest expiry. On an index's own expiry day, that index is also
    run with next week's expiry (reported separately).

Fills: entry/exit at the minute bar's open (if the bar is missing, the last close).
SL is hit when the bar's high >= SL; fill at max(SL, bar open) (gap-up fills worse).
Costs: brokerage 0, statutory charges (see CHARGES), plus --slippage-ticks adverse
ticks on every fill (sell fills lower, buy fills higher; default 0).
Prices in the DB are integer paise.

Usage: python backtests/straddle_920.py [--db data/upstox_options.db] [--out results/straddle_920]
"""

import argparse
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

IST = timezone(timedelta(hours=5, minutes=30))
DAY_START = 9 * 60 + 15                      # 09:15 -> minute index 0
N_MIN = (15 * 60 + 40) - DAY_START           # 09:15 .. 15:39
ENTRY_FIRST, ENTRY_LAST = "09:20", "11:00"
REENTRY_LAST = "14:50"
EXIT_AT = "15:35"
SL_PCT = 0.20
TICK = 0.05                                  # NIFTY / SENSEX option tick size (Rs)

# Statutory charges on option premium turnover (brokerage = 0).
CHARGES = {
    "NIFTY": {"txn": 0.03503 / 100},         # NSE options exchange txn charge
    "SENSEX": {"txn": 0.0325 / 100},         # BSE SENSEX options exchange txn charge
}
STT_SELL = 0.10 / 100                        # STT on option sell premium
SEBI = 10 / 1e7                              # Rs 10 per crore
STAMP_BUY = 0.003 / 100                      # stamp duty on buy side
GST = 0.18                                   # on (txn + SEBI) since brokerage is 0


def mi(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m - DAY_START


def hhmm(i):
    t = DAY_START + i
    return f"{t // 60:02d}:{t % 60:02d}"


def charges(index, sell_value, buy_value):
    turnover = sell_value + buy_value
    txn = turnover * CHARGES[index]["txn"]
    sebi = turnover * SEBI
    return sell_value * STT_SELL + buy_value * STAMP_BUY + txn + sebi + GST * (txn + sebi)


class DayBook:
    """All strikes of one (index, expiry) for one trading day on a 1-minute grid."""

    def __init__(self, db, index, expiry, day):
        start = int(datetime.fromisoformat(day + "T09:15:00").replace(tzinfo=IST).timestamp())
        rows = db.execute(
            "SELECT c.strike, c.option_type, o.ts, o.open, o.high, o.close"
            " FROM option_candles o JOIN contracts c ON c.id = o.contract_id"
            " WHERE c.underlying=? AND c.expiry=? AND o.ts >= ? AND o.ts < ?",
            (index, expiry, start, start + N_MIN * 60)).fetchall()
        self.strikes = np.array(sorted({r[0] for r in rows}))
        pos = {k: i for i, k in enumerate(self.strikes)}
        shape = (2, len(self.strikes), N_MIN)
        self.open = np.full(shape, np.nan)
        self.high = np.full(shape, np.nan)
        close = np.full(shape, np.nan)
        for k, ot, ts, o, h, c in rows:
            j, m = (0 if ot == "CE" else 1), (ts - start) // 60
            self.open[j, pos[k], m] = o / 100
            self.high[j, pos[k], m] = h / 100
            close[j, pos[k], m] = c / 100
        # last known close (forward filled) for minutes without a bar
        self.ffill = close.copy()
        for m in range(1, N_MIN):
            gap = np.isnan(self.ffill[:, :, m])
            self.ffill[:, :, m][gap] = self.ffill[:, :, m - 1][gap]
        self.lot = db.execute("SELECT lot_size FROM contracts WHERE underlying=? AND expiry=?"
                              " LIMIT 1", (index, expiry)).fetchone()[0]

    def price(self, j, s, m):
        """Tradeable price at the start of minute m: bar open, else last close."""
        o = self.open[j, s, m]
        if not np.isnan(o):
            return o
        return self.ffill[j, s, m - 1] if m > 0 else np.nan

    def atm(self, m):
        ce = np.array([self.price(0, s, m) for s in range(len(self.strikes))])
        pe = np.array([self.price(1, s, m) for s in range(len(self.strikes))])
        ok = ~np.isnan(ce) & ~np.isnan(pe)
        if not ok.any():
            return None, None
        diff = np.where(ok, ce - pe, np.inf)
        s = int(np.argmin(np.abs(diff)))
        fut = self.strikes[s] + diff[s]
        cand = np.where(ok)[0]
        return int(cand[np.argmin(np.abs(self.strikes[cand] - fut))]), fut


def run_straddle(book, m0, m_exit):
    """One straddle from minute m0. Returns (legs, both_stopped, minute_after_last_stop)."""
    s, fut = book.atm(m0)
    if s is None:
        return None
    legs = []
    for j in (0, 1):
        e = book.price(j, s, m0)
        legs.append({"type": "CE" if j == 0 else "PE", "strike": float(book.strikes[s]),
                     "entry": e, "sl": e * (1 + SL_PCT), "exit": None, "exit_m": None,
                     "reason": None})
    if any(np.isnan(l["entry"]) for l in legs):
        return None
    for m in range(m0, m_exit):
        hit_now = []
        for j, l in enumerate(legs):
            if l["exit"] is not None:
                continue
            h = book.high[j, s, m]
            if not np.isnan(h) and h >= l["sl"]:
                o = book.open[j, s, m]
                l["exit"] = max(l["sl"], o) if (not np.isnan(o) and m > m0) else l["sl"]
                l["exit_m"], l["reason"] = m, "SL" if l["sl"] > l["entry"] else "SL@cost"
                hit_now.append(j)
        if hit_now and len(hit_now) == 1:
            other = legs[1 - hit_now[0]]
            if other["exit"] is None:
                other["sl"] = other["entry"]       # cost-to-cost from next minute
        if all(l["exit"] is not None for l in legs):
            return legs, fut, True, m + 1
    for j, l in enumerate(legs):
        if l["exit"] is None:
            l["exit"], l["exit_m"], l["reason"] = book.price(j, s, m_exit), m_exit, "time"
    return legs, fut, False, None


def simulate_entry(book, index, m_entry, slip=0.0):
    m_exit, m_re_last = mi(EXIT_AT), mi(REENTRY_LAST)
    trades, m0 = [], m_entry
    for attempt in (0, 1):
        res = run_straddle(book, m0, m_exit)
        if res is None:
            break
        legs, fut, both, m_next = res
        for l in legs:
            sell = max(l["entry"] - slip, 0) * book.lot
            buy = (l["exit"] + slip) * book.lot
            l.update(attempt=attempt, entry_time=hhmm(m0), exit_time=hhmm(l["exit_m"]),
                     synthetic=fut, gross=sell - buy, charges=charges(index, sell, buy))
            l["net"] = l["gross"] - l["charges"]
            trades.append(l)
        if not both or attempt == 1 or m_next is None or m_next > m_re_last:
            break
        m0 = m_next
    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/upstox_options.db")
    ap.add_argument("--out", default="results/straddle_920")
    ap.add_argument("--month", default="2026-10", help="trade dates to test (YYYY-MM)")
    ap.add_argument("--slippage-ticks", type=int, default=0,
                    help="adverse ticks per fill, each leg entry and exit (tick = Rs 0.05)")
    a = ap.parse_args()
    db = sqlite3.connect(a.db)
    os.makedirs(a.out, exist_ok=True)

    entries = range(mi(ENTRY_FIRST), mi(ENTRY_LAST) + 1)
    all_trades = []
    for index in ("NIFTY", "SENSEX"):
        expiries = [r[0] for r in db.execute(
            "SELECT DISTINCT expiry FROM contracts WHERE underlying=? AND candles_done=1"
            " ORDER BY expiry", (index,))]
        days = [r[0] for r in db.execute(
            "SELECT DISTINCT date(o.ts,'unixepoch','+330 minutes') FROM option_candles o"
            " JOIN contracts c ON c.id=o.contract_id WHERE c.underlying=? AND"
            " date(o.ts,'unixepoch','+330 minutes') LIKE ? ORDER BY 1", (index, a.month + "%"))]
        for day in days:
            nearest = [e for e in expiries if e >= day]
            if not nearest:
                print(f"{index} {day}: no downloaded expiry, skipped")
                continue
            runs = [("nearest", nearest[0])]
            if nearest[0] == day:
                if len(nearest) > 1:
                    runs.append(("expiry_day_next_week", nearest[1]))
                else:
                    print(f"{index} {day}: expiry day, next-week contracts not downloaded")
            for variant, expiry in runs:
                book = DayBook(db, index, expiry, day)
                if not len(book.strikes):
                    continue
                for m in entries:
                    for t in simulate_entry(book, index, m, a.slippage_ticks * TICK):
                        t.update(index=index, day=day, expiry=expiry, variant=variant,
                                 slot=hhmm(m))
                        all_trades.append(t)
            print(f"{index} {day}: {', '.join(f'{v} {e}' for v, e in runs)}")

    cols = ["index", "variant", "day", "expiry", "slot", "attempt", "type", "strike",
            "synthetic", "entry_time", "entry", "exit_time", "exit", "reason", "gross",
            "charges", "net"]
    with open(os.path.join(a.out, "trades.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for t in all_trades:
            f.write(",".join(f"{t[c]:.2f}" if isinstance(t[c], float) else str(t[c])
                             for c in cols) + "\n")

    report = []
    groups = defaultdict(list)
    for t in all_trades:
        groups[(t["index"], t["variant"])].append(t)
    for (index, variant), ts in sorted(groups.items()):
        daily = defaultdict(float)
        slot = defaultdict(float)
        straddles = defaultdict(float)
        for t in ts:
            daily[t["day"]] += t["net"]
            slot[t["slot"]] += t["net"]
            straddles[(t["day"], t["slot"])] += t["net"]
        eq = np.cumsum([daily[d] for d in sorted(daily)])
        dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq)) if len(eq) else 0
        re = sum(1 for t in ts if t["attempt"] == 1) // 2
        wins = sum(1 for v in straddles.values() if v > 0)
        report.append({
            "index": index, "variant": variant, "days": len(daily),
            "straddles": len(straddles), "reentries": re,
            "gross": sum(t["gross"] for t in ts), "charges": sum(t["charges"] for t in ts),
            "net": sum(t["net"] for t in ts),
            "win_days": sum(1 for v in daily.values() if v > 0),
            "straddle_win_rate": wins / len(straddles) if straddles else 0,
            "max_dd": dd, "best_day": max(daily.values()), "worst_day": min(daily.values()),
            "daily": dict(sorted(daily.items())), "slot": dict(sorted(slot.items())),
        })
        with open(os.path.join(a.out, f"daily_{index}_{variant}.csv"), "w") as f:
            f.write("day,net\n" + "".join(f"{d},{v:.2f}\n" for d, v in sorted(daily.items())))
        with open(os.path.join(a.out, f"slot_{index}_{variant}.csv"), "w") as f:
            f.write("slot,net\n" + "".join(f"{s},{v:.2f}\n" for s, v in sorted(slot.items())))

    for r in report:
        print(f"\n== {r['index']} / {r['variant']}: {r['days']} days, {r['straddles']} straddles,"
              f" {r['reentries']} re-entries")
        print(f"   gross {r['gross']:,.0f}  charges {r['charges']:,.0f}  NET {r['net']:,.0f}")
        print(f"   win days {r['win_days']}/{r['days']}  straddle win% {r['straddle_win_rate']:.1%}"
              f"  maxDD {r['max_dd']:,.0f}  best {r['best_day']:,.0f}  worst {r['worst_day']:,.0f}")
        for d, v in r["daily"].items():
            print(f"     {d}  {v:>12,.0f}")
    import json
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)


if __name__ == "__main__":
    main()
