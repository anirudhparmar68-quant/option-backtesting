"""QuantStats tearsheets for every strategy built in this project, in one folder.

Capital per strategy = rungs (positions open at the same time) x Rs 1.5 lakh margin per
rung (owner's figure; used for flies/condors/calendars too, which is conservative since
hedged positions need less margin). Daily P&L / capital = daily return (QuantStats then
compounds). Benchmark: NIFTY index.

Output: reports/<name>.html, reports/<name>_metrics.csv, reports/<name>_daily_pnl.csv,
reports/summary.csv. Usage: python backtests/all_reports.py
"""
import collections
import json
import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import quantstats as qs

import straddle_fast as sf

warnings.filterwarnings("ignore")
MARGIN = 1.5e5
OUT = "reports"
HOLDOUT = "2026-07-01"

# ---------------------------------------------------------------- staggered straddles
STRADDLE_CFG = {
    "A_straddle_0920_single": dict(p=dict(sl=0.2, cost=1.0, max_re=1, slip=0), entries=(5, 5)),
    "B_straddle_staggered_plain": dict(p=dict(sl=0.2, cost=1.0, max_re=1, slip=0), entries=(5, 225)),
    "D_CCS_close_confirmed": dict(p=dict(sl=0.2, cost=1.0, max_re=1, slip=1, trig="close"), entries=(5, 225)),
    "F_straddle_index_0p2pct_SL": dict(p=dict(sl=0.2, mv=0.002, cost=1.0, max_re=1, slip=0, trig="index"),
                                       entries=(5, 225)),
}
DAYS = None


def _init():
    global DAYS
    DAYS = sf.attach_spot(sf.load(parallel=False))


def _one(i):
    d = DAYS[i]
    out = {}
    for k, c in STRADDLE_CFG.items():
        E = np.arange(c["entries"][0], c["entries"][1] + 1)
        out[k] = float(sf.entry_pnl(d, c["p"], E).sum())
    return i, out


def straddle_series():
    days = sf.attach_spot(sf.load())
    res = [None] * len(days)
    with Pool(4, initializer=_init) as pool:
        for i, r in pool.imap_unordered(_one, range(len(days)), chunksize=4):
            res[i] = r
    bars = json.load(open("data/daily_bars.json"))
    gap = []
    for d in days:
        b = bars[d["index"]]
        ds = sorted(b)
        k = ds.index(d["day"])
        gap.append(abs(b[d["day"]]["open"] / b[ds[k - 1]]["close"] - 1) > 0.01)
    S = collections.defaultdict(lambda: collections.defaultdict(float))
    for i, d in enumerate(days):
        for k in STRADDLE_CFG:
            S[k][d["day"]] += res[i][k]
        S["C_straddle_staggered_gapfilter"][d["day"]] += 0.0 if gap[i] else res[i]["B_straddle_staggered_plain"]
    # CCS uses the gap filter
    S["D_CCS_close_confirmed"] = collections.defaultdict(float)
    for i, d in enumerate(days):
        if not gap[i]:
            S["D_CCS_close_confirmed"][d["day"]] += res[i]["D_CCS_close_confirmed"]
    return {k: pd.Series(v).sort_index() for k, v in S.items()}


def vol_target(s):
    import final_strategy as fs
    return pd.Series(fs.vol_target(s.values), index=s.index)


# ---------------------------------------------------------------- others
def strangle_series():
    import strangle_100 as s1
    days = s1._days(True)
    out = {}
    with Pool(4, initializer=s1._init) as pool:
        res = dict(pool.map(s1._one, range(len(days)), chunksize=4))
    for name, key in (("G_strangle_100_plain", "plain (SL price, 0 tick)"),
                      ("H_strangle_100_realistic", "realistic (close-confirmed, 1 tick)")):
        out[name] = pd.Series({d["day"]: res[i][key] for i, d in enumerate(days)}).sort_index()
    return out


def fly_condor_series():
    import ironcondor_s2 as c2
    import ironfly_s1 as s
    days = s.load_cache()
    out = {}
    d1, _, _ = s.run_all(days, dict(s.DEFAULT, slip_ticks=3))
    out["I_S1_ironfly_rules_3tick"] = pd.Series(d1).sort_index()
    d2, _, _ = s.run_all(days, dict(s.DEFAULT, slip_ticks=3, wing=0.07, reentry=False))
    out["J_S1_ironfly_wing7_noRe_3tick"] = pd.Series(d2).sort_index()
    out["K_S2_ironcondor_rules_3tick"] = pd.Series(c2.run_all(days, dict(c2.DEFAULT, slip_ticks=3))).sort_index()
    return out


def calendar_series():
    import calendar_cache as cc
    import call_calendar_ladder as cl
    days, spot = cc.load(), cl.load_spot()
    out, rungs = {}, {}
    for name, kw in (("L_calendar_ladder_original", dict(step=100.0, exit_at="15:15", max_dte=None)),
                     ("M_calendar_ladder_tuned", dict(step=100.0, exit_at="09:30", max_dte=3))):
        daily, log = cl.run(days, spot, 1, **kw)
        out[name] = pd.Series(daily).sort_index()
        rungs[name] = max(c[1] for c in log["per_cycle"])
    return out, rungs


def report(name, pnl, capital, bench, title):
    pnl.index = pd.to_datetime(pnl.index)
    rets = (pnl / capital).astype(float)
    b = bench.reindex(rets.index).fillna(0.0)
    pnl.to_csv(f"{OUT}/{name}_daily_pnl.csv", header=["pnl"])
    qs.reports.html(rets, benchmark=b, rf=0.0, title=title, output=f"{OUT}/{name}.html", periods_per_year=250)
    m = qs.reports.metrics(rets, benchmark=b, rf=0.0, mode="full", display=False, periods_per_year=250)
    m.to_csv(f"{OUT}/{name}_metrics.csv")
    v = pnl.values
    eq = np.cumsum(v)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq))
    ho = pnl[pnl.index >= HOLDOUT]
    mon = pnl.groupby(pnl.index.to_period("M")).sum()
    col = m.columns[-1]
    get = lambda k: m.loc[k, col] if k in m.index else None
    return dict(strategy=name, capital_cr=round(capital / 1e7, 3), net_rs=round(float(v.sum())), max_dd_rs=round(dd),
                ret_dd=round(float(v.sum()) / dd, 2) if dd else None,
                cagr=get("CAGR﹪"), max_dd_pct=get("Max Drawdown"), sharpe=get("Sharpe"), sortino=get("Sortino"),
                calmar=get("Calmar"), win_days=get("Win Days"), win_months=get("Win Month"),
                worst_day_pct=get("Worst Day"), loss_months=int((mon < 0).sum()), months=len(mon),
                holdout_net_rs=round(float(ho.sum())),
                holdout_sharpe=round(float(ho.mean() / ho.std() * np.sqrt(250)), 2) if ho.std() > 0 else None)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    bars = json.load(open("data/daily_bars.json"))["NIFTY"]
    bench = pd.Series({pd.Timestamp(k): v["close"] for k, v in bars.items()}).sort_index().pct_change()
    S = straddle_series()
    S["E_CCS_vol_target"] = vol_target(S["D_CCS_close_confirmed"])
    S.update(strangle_series())
    S.update(fly_condor_series())
    cal, cal_rungs = calendar_series()
    S.update(cal)
    TITLES = {
        "A_straddle_0920_single": ("Single 09:20 straddle, NIFTY+SENSEX, SL price, 0 tick", 2),
        "B_straddle_staggered_plain": ("Staggered straddle 09:20-13:00, NIFTY+SENSEX, SL price, 0 tick", 442),
        "C_straddle_staggered_gapfilter": ("Staggered straddle + skip gap>1%, SL price, 0 tick", 442),
        "D_CCS_close_confirmed": ("CCS: close-confirmed stops, next-minute fills, 1 tick, gap filter", 442),
        "E_CCS_vol_target": ("CCS + volatility targeting (0.5x-1.5x)", 442),
        "F_straddle_index_0p2pct_SL": ("Staggered straddle, 0.2% index-move SL, 0 tick", 442),
        "G_strangle_100_plain": ("NIFTY Rs100 strangle 09:20-13:00, SL price, 0 tick", 221),
        "H_strangle_100_realistic": ("NIFTY Rs100 strangle, close-confirmed, 1 tick", 221),
        "I_S1_ironfly_rules_3tick": ("S1 NIFTY iron fly (user rules), 3 ticks", 221),
        "J_S1_ironfly_wing7_noRe_3tick": ("S1 iron fly, 7-delta wings, no re-entry, 3 ticks", 221),
        "K_S2_ironcondor_rules_3tick": ("S2 NIFTY iron condor (user rules), 3 ticks", 221),
        "L_calendar_ladder_original": ("NIFTY call-calendar ladder (user idea), 1 tick",
                                       cal_rungs["L_calendar_ladder_original"]),
        "M_calendar_ladder_tuned": ("Calendar ladder: exit 09:30 on expiry, last 3 days, 1 tick",
                                    cal_rungs["M_calendar_ladder_tuned"]),
    }
    rows = []
    for name in sorted(TITLES):
        title, rungs = TITLES[name]
        rows.append(dict(report(name, S[name].copy(), rungs * MARGIN, bench, title), rungs=rungs, title=title))
        print("done", name, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/summary.csv", index=False)
    print(pd.DataFrame(rows)[["strategy", "rungs", "capital_cr", "net_rs", "max_dd_rs", "ret_dd", "cagr",
                              "max_dd_pct", "sharpe", "sortino", "calmar", "loss_months", "holdout_sharpe"]].to_string())
