"""Staggered straddle with an underlying-move stop (user rules): every minute 09:20-13:00,
leg stop on a 0.2% index move (CE: index up, PE: index down, 1-min close, exit next
minute), survivor at cost, 1 re-entry (not after 14:50), CAS-aware exits, charges.
Usage: python backtests/straddle_index_sl.py"""
import collections
import json
from multiprocessing import Pool

import numpy as np

import straddle_fast as sf

CFG = {"index 0.2%, 0 tick": dict(sl=0.2, mv=0.002, cost=1.0, max_re=1, slip=0, trig="index"),
       "index 0.2%, 1 tick": dict(sl=0.2, mv=0.002, cost=1.0, max_re=1, slip=1, trig="index"),
       "premium 20% (same execution), 0 tick": dict(sl=0.2, cost=1.0, max_re=1, slip=0, trig="close")}
DAYS = None


def _init():
    global DAYS
    DAYS = sf.attach_spot(sf.load(parallel=False))


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
    days = sf.attach_spot(sf.load())
    res = [None] * len(days)
    with Pool(4, initializer=_init) as pool:
        for i, r in pool.imap_unordered(_one, range(len(days)), chunksize=4):
            res[i] = r
    dates = sorted({d["day"] for d in days})
    pos = {x: i for i, x in enumerate(dates)}
    out = {}
    for k in CFG:
        ser = {ix: np.zeros(len(dates)) for ix in ("NIFTY", "SENSEX")}
        for i, d in enumerate(days):
            ser[d["index"]][pos[d["day"]]] += res[i][k]
        tot = ser["NIFTY"] + ser["SENSEX"]
        mon = collections.defaultdict(lambda: [0.0, 0.0])
        for x, a, b in zip(dates, ser["NIFTY"], ser["SENSEX"]):
            mon[x[:7]][0] += a
            mon[x[:7]][1] += b
        out[k] = dict(NIFTY=metrics(ser["NIFTY"]), SENSEX=metrics(ser["SENSEX"]), BOTH=metrics(tot),
                      months={m: [round(a), round(b)] for m, (a, b) in sorted(mon.items())})
        print(k, json.dumps({x: out[k][x] for x in ("NIFTY", "SENSEX", "BOTH")}))
    json.dump(out, open("results/straddle_index_sl.json", "w"), indent=1)
    for m, (a, b) in out["index 0.2%, 0 tick"]["months"].items():
        print("M", m, a, b, a + b)
