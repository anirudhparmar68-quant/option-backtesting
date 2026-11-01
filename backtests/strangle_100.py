"""NIFTY 'sell ~Rs100 premium' staggered strangle (user rules).

Every minute 09:20-13:00: sell the CE and the PE whose price is nearest Rs 100 (nearest
expiry; on NIFTY's own expiry day the NEXT expiry). Leg SL 20% of premium, survivor at
cost, 1 re-entry (fresh Rs100 strikes, not after 14:50). Exit 15:15 pre-CAS and on expiry
days, 15:35 otherwise. Charges; fills: 'plain' = SL price (no slippage) and 'realistic' =
close-confirmed, next-minute execution, 1 tick.
Usage: python backtests/strangle_100.py"""
import collections
import json
from multiprocessing import Pool

import numpy as np

import straddle_fast as sf

CFG = {"plain (SL price, 0 tick)": dict(sl=0.2, cost=1.0, max_re=1, slip=0, pick="premium"),
       "realistic (close-confirmed, 1 tick)": dict(sl=0.2, cost=1.0, max_re=1, slip=1, pick="premium", trig="close"),
       "ATM straddle, plain (for comparison)": dict(sl=0.2, cost=1.0, max_re=1, slip=0),
       "ATM straddle, realistic": dict(sl=0.2, cost=1.0, max_re=1, slip=1, trig="close")}
DAYS = None


def _days(parallel):
    return sf.attach_premium_picks(sf.load(index="NIFTY", parallel=parallel, next_on_expiry=True), 100.0)


def _init():
    global DAYS
    DAYS = _days(False)


def _one(i):
    return i, {k: float(sf.entry_pnl(DAYS[i], p).sum()) for k, p in CFG.items()}


def metrics(v):
    v = np.asarray(v, float)
    eq = np.cumsum(v)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq))
    return dict(net=round(float(v.sum())), max_dd=round(dd), ret_dd=round(float(v.sum()) / dd, 2),
                sharpe=round(float(v.mean() / v.std() * np.sqrt(250)), 2),
                sortino=round(float(v.mean() / np.sqrt((np.minimum(v, 0) ** 2).mean()) * np.sqrt(250)), 2),
                win_days=round(float((v > 0).mean()), 3))


if __name__ == "__main__":
    days = _days(True)
    res = [None] * len(days)
    with Pool(4, initializer=_init) as pool:
        for i, r in pool.imap_unordered(_one, range(len(days)), chunksize=4):
            res[i] = r
    order = np.argsort([d["day"] for d in days])
    dates = [days[i]["day"] for i in order]
    out = {}
    for k in CFG:
        v = np.array([res[i][k] for i in order])
        mon = collections.defaultdict(float)
        for x, y in zip(dates, v):
            mon[x[:7]] += y
        exp = np.array([days[i]["day"] == days[i]["expiry"] or days[i]["X"] == 0 for i in order])
        out[k] = dict(full=metrics(v), months={m: round(y) for m, y in sorted(mon.items())},
                      loss_months=sum(1 for y in mon.values() if y < 0))
        print(k, json.dumps(out[k]["full"]), "loss months", out[k]["loss_months"])
    json.dump(out, open("results/strangle_100.json", "w"), indent=1)
    for m in out["plain (SL price, 0 tick)"]["months"]:
        print("M", m, *[out[k]["months"][m] for k in CFG])
