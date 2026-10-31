"""Execution-realism check for the staggered straddle: the user's rules and the two
candidate tweaks under SL-price vs close fills and 0 vs 1 tick of slippage.
Gap = index open / previous close - 1 (09:15); with --gap-at-0920 the index price at
09:20 (open of the 09:20 minute, i.e. known at entry) is used instead of the 09:15 open.
Usage: python backtests/straddle_scenarios.py [--gap-at-0920]"""
import collections
import json
import sys
from multiprocessing import Pool

import numpy as np

import straddle_fast as sf

HOLDOUT = "2026-07-01"
CONFIGS = {"rules (SL20)": dict(sl=0.2, cost=1.0, max_re=1), "SL15": dict(sl=0.15, cost=1.0, max_re=1)}
SCEN = {"SL-price, 0 tick": dict(fill="sl", slip=0), "SL-price, 1 tick": dict(fill="sl", slip=1),
        "close, 0 tick": dict(fill="close", slip=0), "close, 1 tick": dict(fill="close", slip=1)}
DAYS = None


def _init():
    global DAYS
    DAYS = sf.load(parallel=False)


def _one(i):
    d = DAYS[i]
    return i, {(c, s): float(sf.entry_pnl(d, dict(cp, **sp)).sum())
               for c, cp in CONFIGS.items() for s, sp in SCEN.items()}


def stats(v):
    v = np.asarray(v, float)
    eq = np.cumsum(v)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq))
    return dict(net=round(float(v.sum())), maxdd=round(dd), calmar=round(float(v.sum()) / dd, 2) if dd else None,
                win=round(float((v > 0).mean()), 3))


if __name__ == "__main__":
    days = sf.load()
    bars = json.load(open("data/daily_bars.json"))
    at920 = "--gap-at-0920" in sys.argv
    i920 = json.load(open("data/index_0920.json")) if at920 else None
    gap = []
    for d in days:
        b = bars[d["index"]]
        ds = sorted(b)
        k = ds.index(d["day"]) if d["day"] in b else 0
        now = (i920[d["index"]].get(d["day"], {}).get("open") if at920 else b[d["day"]]["open"]) if k else None
        gap.append(abs(now / b[ds[k - 1]]["close"] - 1) if now else 0.0)
    print(f"gap measured at {'09:20' if at920 else '09:15'}; days with |gap| > 1%: {int((gap_arr := np.array(gap) > 0.01).sum())}")
    gap = np.array(gap)
    res = [None] * len(days)
    with Pool(4, initializer=_init) as pool:
        for i, r in pool.imap_unordered(_one, range(len(days)), chunksize=4):
            res[i] = r
    dates = sorted({d["day"] for d in days})
    pos = {x: i for i, x in enumerate(dates)}
    dates = np.array(dates)
    out = {}
    for c in CONFIGS:
        for s in SCEN:
            for gname, keep in (("all days", np.ones(len(days), bool)), ("skip gap>1%", gap <= 0.01)):
                v = np.zeros(len(dates))
                mon = collections.defaultdict(float)
                for i, d in enumerate(days):
                    if keep[i]:
                        v[pos[d["day"]]] += res[i][(c, s)]
                for x, y in zip(dates, v):
                    mon[x[:7]] += y
                key = f"{c} | {s} | {gname}"
                out[key] = dict(full=stats(v), pre=stats(v[dates < HOLDOUT]), holdout=stats(v[dates >= HOLDOUT]),
                                loss_months=sum(1 for y in mon.values() if y < 0),
                                months={m: round(y) for m, y in sorted(mon.items())})
                f, p, h = out[key]["full"], out[key]["pre"], out[key]["holdout"]
                print(f"{key:48} FULL net {f['net']:>12,} dd {f['maxdd']:>11,} cal {f['calmar']} | "
                      f"PRE cal {p['calmar']} | HOLD net {h['net']:>11,} cal {h['calmar']} | lossM {out[key]['loss_months']}")
    json.dump(out, open(f"results/straddle_search/scenarios{'_gap0920' if at920 else ''}.json", "w"), indent=1)
