"""NIFTY call-calendar ladder (positional), the user's idea.

Per weekly cycle (near expiry E1, next expiry E2):
  * A calendar = sell 1 lot E1 call at K, buy 1 lot E2 call at K. No stop, no adjustment.
  * Each trading day of the cycle (not on E1's expiry day): at 09:20 and then every
    minute, K = NIFTY spot (1-minute close) rounded to the nearest 100. If no calendar
    exists at K in this cycle yet, open one at the next minute's option prices.
    A strike is used once per cycle (coming back to it does nothing).
  * Positions are carried overnight. On E1's expiry day everything is closed at 15:15:
    the short is bought back, the long sold.
Fills at next-minute option prices; statutory charges + `slip` ticks per leg per fill.
Daily P&L = realised + change in mark-to-market (marks at each day's last traded price).

Data caveat: data/upstox_7delta.db holds strikes down to ~7 delta only. A calendar whose
strike drifts deep in/out of the money can lose its prices; every such case is counted
(missing_marks / missing_exit) and, for the 0-DTE short at exit, intrinsic value is used.
"""

import collections

import numpy as np

import straddle_920 as st


def load_spot():
    z = np.load("data/nifty_index_1min.npz")
    return {d: oc for d, oc in zip(z["days"], z["oc"])}


def run(days, spot, slip_ticks=1, step=100.0, last_entry="15:10", start="09:20",
        exit_at="15:15", max_dte=None, side="call"):
    """side='call': call calendars everywhere; 'otm': call calendar at/above spot, put
    calendar below spot (the out-of-the-money, more liquid option). max_dte: only open
    new calendars when the near expiry is at most this many calendar days away."""
    days = sorted([d for d in days if d["index"] == "NIFTY"], key=lambda d: d["day"])
    slip = slip_ticks * st.TICK
    cycles = collections.OrderedDict()
    for d in days:
        cycles.setdefault(str(d["exp0"]), []).append(d)
    daily = collections.OrderedDict()
    log = dict(calendars=0, cycles=0, skipped_cycles=0, missing_marks=0, missing_exit=0,
               entry_unavailable=0, per_cycle=[])
    for e1, cd in cycles.items():
        if cd[-1]["day"] != e1:                  # cycle without its expiry day in the data
            log["skipped_cycles"] += 1
            continue
        log["cycles"] += 1
        pos = []                                 # dicts: K, s_px, l_px, mark
        used = set()
        prev_mark = 0.0
        for d in cd:
            oc = spot.get(d["day"])
            k0, k1, lot, X = d["k0"], d["k1"], d["lot"], d["X"]
            i0 = {float(k): i for i, k in enumerate(k0)}
            i1 = {float(k): i for i, k in enumerate(k1)}
            realised = 0.0
            dte = (np.datetime64(e1) - np.datetime64(d["day"])).astype(int)
            if d["day"] != e1 and oc is not None and (max_dte is None or dte <= max_dte):
                for m in range(st.mi(start), min(st.mi(last_entry), X - 1) + 1):
                    c = oc[m, 1]
                    if not np.isfinite(c):
                        continue
                    K = float(np.floor(c / step + 0.5) * step)
                    if K in used:
                        continue
                    if K not in i0 or K not in i1:
                        log["entry_unavailable"] += 1
                        continue
                    j = 1 if (side == "otm" and K < c) else 0      # 0 = CE, 1 = PE
                    sp = round(float(d["px0"][j, i0[K], m + 1]), 2)
                    lp = round(float(d["px1"][j, i1[K], m + 1]), 2)
                    if np.isnan(sp) or np.isnan(lp):
                        log["entry_unavailable"] += 1
                        continue
                    s_fill, l_fill = max(sp - slip, 0), lp + slip
                    realised -= st.charges("NIFTY", s_fill * lot, 0) + st.charges("NIFTY", 0, l_fill * lot)
                    pos.append(dict(K=K, j=j, s=s_fill, l=l_fill, lot=lot, mark=(s_fill, l_fill)))
                    used.add(K)
                    log["calendars"] += 1
            if d["day"] == e1:
                # close everything at the exit minute
                Xe = min(st.mi(exit_at), X)
                sclose = oc[Xe, 0] if oc is not None and np.isfinite(oc[Xe, 0]) else np.nan
                for p in pos:
                    K, j = p["K"], p["j"]
                    sx = round(float(d["px0"][j, i0[K], Xe]), 2) if K in i0 else np.nan
                    lx = round(float(d["px1"][j, i1[K], Xe]), 2) if K in i1 else np.nan
                    if np.isnan(sx):
                        intr = (sclose - K) if j == 0 else (K - sclose)
                        sx = max(intr, 0) if np.isfinite(sclose) else p["mark"][0]
                        log["missing_exit"] += 1
                    if np.isnan(lx):
                        lx = p["mark"][1]
                        log["missing_exit"] += 1
                    buy, sell = (sx + slip) * p["lot"], max(lx - slip, 0) * p["lot"]
                    realised += (p["s"] * p["lot"] - buy) + (sell - p["l"] * p["lot"])
                    realised -= st.charges("NIFTY", 0, buy) + st.charges("NIFTY", sell, 0)
                daily[d["day"]] = realised - prev_mark      # reverse yesterday's MTM
                log["per_cycle"].append((e1, len(pos), round(sum(0 for _ in pos))))
                pos, prev_mark = [], 0.0
                continue
            # mark open positions at the day's last price (forward-filled closes)
            mark = 0.0
            for p in pos:
                K, j = p["K"], p["j"]
                sm = float(d["cl0"][j, i0[K], -1]) if K in i0 else np.nan
                lm = float(d["cl1"][j, i1[K], -1]) if K in i1 else np.nan
                if np.isnan(sm) or np.isnan(lm):
                    log["missing_marks"] += 1
                    sm = p["mark"][0] if np.isnan(sm) else sm
                    lm = p["mark"][1] if np.isnan(lm) else lm
                p["mark"] = (sm, lm)
                mark += ((p["s"] - sm) + (lm - p["l"])) * p["lot"]
            daily[d["day"]] = realised + mark - prev_mark
            prev_mark = mark
    return daily, log


def metrics(v):
    v = np.asarray(v, float)
    eq = np.cumsum(v)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq))
    return dict(net=round(float(v.sum())), max_dd=round(dd), ret_dd=round(float(v.sum()) / dd, 2) if dd else None,
                sharpe=round(float(v.mean() / v.std() * np.sqrt(250)), 2),
                sortino=round(float(v.mean() / np.sqrt((np.minimum(v, 0) ** 2).mean()) * np.sqrt(250)), 2),
                win_days=round(float((v > 0).mean()), 3), days=len(v))


if __name__ == "__main__":
    import calendar_cache as cc
    days = cc.load()
    spot = load_spot()
    for slip in (0, 1, 2):
        daily, log = run(days, spot, slip)
        ds = np.array(list(daily))
        v = np.array(list(daily.values()))
        print(f"slip {slip}: FULL {metrics(v)}")
        print(f"         PRE  {metrics(v[ds < '2026-07-01'])}")
        print(f"         HOLD {metrics(v[ds >= '2026-07-01'])}")
        print("        ", {k: v for k, v in log.items() if k != "per_cycle"},
              "avg calendars/cycle", round(np.mean([c[1] for c in log["per_cycle"]]), 1))
