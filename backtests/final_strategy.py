"""Final candidate: 'Close-confirmed staggered straddle' (CCS), NIFTY + SENSEX.

Rules (each entry independent, 1 lot per leg at base size):
  * Every minute 09:20-13:00, sell the ATM straddle (synthetic ATM, nearest expiry,
    the expiring contract on expiry day).
  * Leg stop 20% over entry, confirmed only by a 1-minute CLOSE, executed at the next
    minute's price. After one leg stops, the other leg's stop moves to cost (same
    close-confirmed logic). Both stopped -> 1 re-entry at that next minute, not after 14:50.
  * Time exit 15:15 (pre-CAS and expiry days) / 15:35 (post-CAS non-expiry days).
  * Skip an index for the day if its open gaps more than 1% from the previous close.
  * Optional overlay: volatility targeting - day size = clip(target / trailing 20-day
    std of the strategy's own daily P&L, 0.5, 1.5), using only past days.
Costs: statutory charges + 1 tick per leg per fill.
Usage: python backtests/final_strategy.py
"""
import collections
import json
from multiprocessing import Pool

import numpy as np

import straddle_fast as sf

P = dict(sl=0.2, cost=1.0, max_re=1, slip=1, trig="close")
HOLDOUT = "2026-07-01"
DAYS = None


def _init():
    global DAYS
    DAYS = sf.load(parallel=False)


def _one(i):
    return i, float(sf.entry_pnl(DAYS[i], P).sum()), float(sf.entry_pnl(DAYS[i], dict(P, slip=2)).sum())


def metrics(v):
    v = np.asarray(v, float)
    eq = np.cumsum(v)
    peak = np.maximum.accumulate(np.concatenate([[0], eq]))[1:]
    dd = float(np.max(peak - eq))
    yrs = len(v) / 250
    return dict(net=round(float(v.sum())), max_dd=round(dd), ret_dd=round(float(v.sum()) / dd, 2),
                calmar_annual=round(float(v.sum()) / yrs / dd, 2),
                sharpe=round(float(v.mean() / v.std() * np.sqrt(250)), 2),
                sortino=round(float(v.mean() / np.sqrt((np.minimum(v, 0) ** 2).mean()) * np.sqrt(250)), 2),
                win_days=round(float((v > 0).mean()), 3), worst_day=round(float(v.min())),
                best_day=round(float(v.max())), days=len(v))


def vol_target(v, lookback=20, lo=0.5, hi=1.5):
    v = np.asarray(v, float)
    out = v.copy()
    for t in range(lookback, len(v)):
        sd = v[t - lookback:t].std()
        tgt = np.median([v[s - lookback:s].std() for s in range(lookback, t + 1)])   # past-only
        out[t] = v[t] * float(np.clip(tgt / sd, lo, hi)) if sd > 0 else v[t]
    return out


if __name__ == "__main__":
    days = sf.load()
    bars = json.load(open("data/daily_bars.json"))
    res = [None] * len(days)
    with Pool(4, initializer=_init) as pool:
        for i, a, b in pool.imap_unordered(_one, range(len(days)), chunksize=4):
            res[i] = (a, b)
    dates = sorted({d["day"] for d in days})
    pos = {x: i for i, x in enumerate(dates)}
    series = {k: np.zeros(len(dates)) for k in ("NIFTY", "SENSEX", "slip2")}
    for i, d in enumerate(days):
        b = bars[d["index"]]
        ds = sorted(b)
        k = ds.index(d["day"])
        if abs(b[d["day"]]["open"] / b[ds[k - 1]]["close"] - 1) > 0.01:
            continue
        series[d["index"]][pos[d["day"]]] += res[i][0]
        series["slip2"][pos[d["day"]]] += res[i][1]
    tot = series["NIFTY"] + series["SENSEX"]
    vt = vol_target(tot)
    dates = np.array(dates)
    pre, ho = dates < HOLDOUT, dates >= HOLDOUT
    report = {}
    for name, v in (("CCS", tot), ("CCS + vol target", vt), ("CCS, 2 ticks", series["slip2"]),
                    ("NIFTY only", series["NIFTY"]), ("SENSEX only", series["SENSEX"])):
        report[name] = dict(full=metrics(v), pre_holdout=metrics(v[pre]), holdout=metrics(v[ho]))
    mon = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for x, a, b, c in zip(dates, series["NIFTY"], series["SENSEX"], vt):
        mon[x[:7]][0] += a
        mon[x[:7]][1] += b
        mon[x[:7]][2] += a + b
        mon[x[:7]][3] += c
    report["monthly"] = {m: [round(y) for y in v] for m, v in sorted(mon.items())}
    worst = np.argsort(tot)[:10]
    report["worst_days"] = [(str(dates[i]), round(float(series["NIFTY"][i])), round(float(series["SENSEX"][i])),
                             round(float(tot[i]))) for i in worst]
    eq = np.cumsum(tot)
    j = int(np.argmax(np.maximum.accumulate(eq) - eq))
    k = int(np.argmax(eq[:j + 1]))
    report["max_dd_period"] = [str(dates[k]), str(dates[j])]
    json.dump(report, open("results/final_strategy.json", "w"), indent=1)
    for name in ("CCS", "CCS + vol target", "CCS, 2 ticks", "NIFTY only", "SENSEX only"):
        print(name, json.dumps(report[name]))
    print("maxDD period", report["max_dd_period"])
    print("worst", report["worst_days"])
    for m, v in report["monthly"].items():
        print("M", m, *v)
