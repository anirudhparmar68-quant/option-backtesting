"""QuantStats tearsheet + Monte Carlo for the NIFTY Rs100-premium strangle (realistic:
close-confirmed stops, next-minute fills, 1 tick, charges).
Daily P&L is turned into returns on a fixed capital base (CAPITAL, no compounding).
Benchmark: NIFTY index daily returns. Usage: python backtests/quantstats_report.py"""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import quantstats as qs

import strangle_100 as s1

CAPITAL = 1e7            # Rs 1 crore notional capital base (assumption; set to your margin)
P = s1.CFG["realistic (close-confirmed, 1 tick)"]


def _one(i):
    import straddle_fast as sf
    return i, float(sf.entry_pnl(s1.DAYS[i], P).sum())


if __name__ == "__main__":
    days = s1._days(True)
    res = [0.0] * len(days)
    with Pool(4, initializer=s1._init) as pool:
        for i, v in pool.imap_unordered(_one, range(len(days)), chunksize=4):
            res[i] = v
    pnl = pd.Series({pd.Timestamp(d["day"]): r for d, r in zip(days, res)}).sort_index()
    rets = pnl / CAPITAL
    bars = json.load(open("data/daily_bars.json"))["NIFTY"]
    bench = pd.Series({pd.Timestamp(k): v["close"] for k, v in bars.items()}).sort_index().pct_change()
    bench = bench.reindex(rets.index).fillna(0.0)
    pnl.to_csv("results/strangle_100_daily_pnl.csv", header=["pnl"])
    qs.reports.html(rets, benchmark=bench, rf=0.0, title="NIFTY Rs100 strangle (realistic)",
                    output="results/strangle_100_tearsheet.html", periods_per_year=250)
    m = qs.reports.metrics(rets, benchmark=bench, rf=0.0, mode="full", display=False, periods_per_year=250)
    m.to_csv("results/strangle_100_quantstats_metrics.csv")
    mc = qs.stats.montecarlo(rets, sims=5000, bust=-0.30, goal=1.0, seed=42)
    print(m.to_string())
    print("\nMonte Carlo (5000 reshuffles):")
    for k in ("bust_probability", "goal_probability"):
        print(k, getattr(mc, k, None))
    try:
        print(mc.stats if not callable(getattr(mc, "stats", None)) else mc.stats())
    except Exception as e:
        print("mc stats:", e)
    dd = qs.stats.montecarlo_drawdown(rets, sims=5000, seed=42)
    print("MC max drawdown distribution:", dd if not hasattr(dd, "describe") else dd.describe())
