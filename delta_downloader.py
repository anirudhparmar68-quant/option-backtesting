"""Download NIFTY / SENSEX expired options, but only what a near/next-expiry
backtest needs:

  * For each trading day only the NEAR and NEXT expiry count (on an expiry day:
    that day's expiry + next week's).
  * Only strikes whose OTM option reaches |delta| >= --min-delta (default 0.07)
    on some of those days (so the range is: 7-delta put strike .. 7-delta call strike,
    CE and PE both kept for every strike in it).
  * Only the 1-minute bars of the days on which the expiry was near/next are stored.

Strikes are chosen BEFORE downloading, from daily data: Black-76 delta with spot =
the day's index high / low, vol = India VIX day-high x --iv-mult (default 1.25, a
cushion for skew), time = 09:15 that day to 15:30 on expiry. A strike is kept if at
some point in the day's range its call delta lies within [min_delta, 1 - min_delta].

Usage:
  python delta_downloader.py                     # both indices, all expiries
  python delta_downloader.py --index NIFTY --expiry-from 2025-01-01
  python delta_downloader.py --plan-only         # just print how many contracts
"""

import argparse
import math
import os
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np

import upstox_downloader as u

VIX_KEY = "NSE_INDEX|India VIX"


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def call_delta(spot, k, vol, t):
    d1 = (math.log(spot / k) + 0.5 * vol * vol * t) / (vol * math.sqrt(t))
    return ncdf(d1)


def daily_bars(api, key, frm, to):
    path = f"/v3/historical-candle/{u.quote(key, safe='')}/days/1/{to}/{frm}"
    data = api.get(path) or {}
    out = {}
    for c in data.get("candles") or []:
        out[c[0][:10]] = {"open": c[1], "high": c[2], "low": c[3], "close": c[4]}
    return out


def near_next_days(trading_days, expiries):
    """{expiry: [days on which it is the near or next expiry]}"""
    out = {}
    for d in trading_days:
        upcoming = [e for e in expiries if e >= d][:2]
        for e in upcoming:
            out.setdefault(e, []).append(d)
    return out


def strike_kept(k, days, idx, vix, expiry, min_delta, iv_mult):
    exp_ts = datetime.fromisoformat(expiry + "T15:30:00")
    for d in days:
        if d not in idx or d not in vix:
            continue
        t = (exp_ts - datetime.fromisoformat(d + "T09:15:00")).total_seconds() / (365 * 86400)
        vol = vix[d]["high"] / 100 * iv_mult
        if (call_delta(idx[d]["high"], k, vol, t) >= min_delta
                and call_delta(idx[d]["low"], k, vol, t) <= 1 - min_delta):
            return True
    return False


def keep_strikes(strikes, days, idx, vix, expiry, a):
    """Strikes inside the delta band, widened by --buffer-pct of spot on each side
    (expiry-day IVs run well above VIX, so the pure VIX band can be too narrow)."""
    core = [k for k in strikes if strike_kept(k, days, idx, vix, expiry, a.min_delta, a.iv_mult)]
    if not core:
        return set()
    buf = a.buffer_pct / 100 * max(idx[d]["high"] for d in days if d in idx)
    return {k for k in strikes if min(core) - buf <= k <= max(core) + buf}


def day_bounds(days):
    """Epoch range [start of first day, end of last day] in IST."""
    first = datetime.fromisoformat(days[0] + "T00:00:00").replace(tzinfo=u_ist())
    last = datetime.fromisoformat(days[-1] + "T23:59:59").replace(tzinfo=u_ist())
    return int(first.timestamp()), int(last.timestamp())


def u_ist():
    from datetime import timezone
    return timezone(timedelta(hours=5, minutes=30))


def ist_day(epoch):
    return datetime.fromtimestamp(epoch, u_ist()).strftime("%Y-%m-%d")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", choices=list(u.INDICES), action="append")
    p.add_argument("--db", default="data/upstox_7delta.db")
    p.add_argument("--expiry-from")
    p.add_argument("--expiry-to")
    p.add_argument("--min-delta", type=float, default=0.07)
    p.add_argument("--iv-mult", type=float, default=1.25)
    p.add_argument("--buffer-pct", type=float, default=1.0,
                   help="widen the strike band by this %% of spot on each side")
    p.add_argument("--per-min", type=int, default=150)
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()

    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        sys.exit("Set UPSTOX_ACCESS_TOKEN env var first.")
    os.makedirs(os.path.dirname(a.db) or ".", exist_ok=True)
    state = os.path.join(os.path.dirname(a.db) or ".", ".upstox_ratelimit.json")
    api = u.Upstox(token, u.RateLimiter(a.per_min, state))
    db = u.open_db(a.db)

    for name in a.index or list(u.INDICES):
        ukey = u.INDICES[name]
        expiries = sorted(api.expiries(ukey))
        if not expiries:
            continue
        frm = (date.fromisoformat(expiries[0]) - timedelta(days=30)).isoformat()
        to = min(date.fromisoformat(expiries[-1]), date.today()).isoformat()
        idx = daily_bars(api, ukey, frm, to)
        vix = daily_bars(api, VIX_KEY, frm, to)
        # before the first listed expiry the true near expiry is unknown (older
        # expiries are not served), so start 6 days before it
        first_ok = (date.fromisoformat(expiries[0]) - timedelta(days=6)).isoformat()
        trading_days = [d for d in sorted(idx) if d >= first_ok]
        windows = near_next_days(trading_days, expiries)
        todo_exp = [e for e in expiries
                    if (not a.expiry_from or e >= a.expiry_from)
                    and (not a.expiry_to or e <= a.expiry_to) and e in windows]
        print(f"{name}: {len(todo_exp)} expiries ({todo_exp[0]} .. {todo_exp[-1]})")

        plan = 0
        for e in todo_exp:
            u.sync_contracts(api, db, name, ukey, e)
            days = windows[e]
            rows = db.execute("SELECT id, instrument_key, strike, candles_done FROM contracts"
                              " WHERE underlying=? AND expiry=?", (name, e)).fetchall()
            keep = keep_strikes({r[2] for r in rows}, days, idx, vix, e, a)
            todo = [r for r in rows if r[2] in keep and not r[3]]
            plan += len(todo)
            print(f"  {name} {e}: near/next on {len(days)} days, keep {len(keep)} strikes"
                  f" ({min(keep) if keep else '-'}..{max(keep) if keep else '-'}),"
                  f" {len(todo)} contracts to fetch")
            if a.plan_only:
                continue
            lo_ts, hi_ts = day_bounds(days)
            for cid, key, k, _ in todo:
                candles = api.expired_candles(key, days[0], e)
                rows_in = [r for r in (u.candle_row(c) for c in candles)
                           if lo_ts <= r[0] <= hi_ts and ist_day(r[0]) in days]
                db.executemany("INSERT OR REPLACE INTO option_candles VALUES (?,?,?,?,?,?,?,?)",
                               [(cid, *r) for r in rows_in])
                db.execute("UPDATE contracts SET candles_done=1, n_candles=? WHERE id=?",
                           (len(rows_in), cid))
                db.commit()
        print(f"{name}: {plan} contracts {'planned' if a.plan_only else 'fetched'}")
    db.close()


if __name__ == "__main__":
    main()
