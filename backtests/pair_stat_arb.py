"""NIFTY vs SENSEX intraday pair trade with option-built synthetic futures.

Signal (minute m, using only prices up to the start of minute m):
  s = ln(F_nifty) - ln(F_sensex)  (synthetic futures, nearest expiry)
  z = (s - rolling mean_L) / rolling std_L over the last L minutes of the day
Enter when |z| >= zin AND |s - mean| >= min_bps: sell the rich index's synthetic
future (sell ATM CE + buy ATM PE) and buy the cheap one (buy CE + sell PE), notional
matched (SENSEX quantity scaled to NIFTY notional). Exit when z crosses back through
zout, after max_hold minutes, or at 15:10. One position at a time; may re-enter.
All orders execute at the NEXT minute's option prices; P&L is from those option prices.
Costs: statutory charges + `slip` ticks per leg per fill.
"""

import numpy as np

import straddle_920 as st

T_START, T_END = st.mi("09:35"), st.mi("15:10")


def leg_cost(index, sell, buy):
    return st.charges(index, sell, buy)


def day_trades(n, s, p):
    """n, s: cached day dicts (straddle_fast) for NIFTY and SENSEX on the same date."""
    fN, fS = n["fut"], s["fut"]
    sp = np.log(fN) - np.log(fS)
    L, zin, zout, mb, hold = p["L"], p["zin"], p["zout"], p["min_bps"] / 1e4, p["max_hold"]
    slip = p.get("slip", 1) * st.TICK
    trades = []
    pos = None
    for m in range(max(T_START, st.mi("09:20") + L), T_END):
        w = sp[m - L:m + 1]
        if not np.isfinite(w).all():
            continue
        mu, sd = w[:-1].mean(), w[:-1].std()
        if sd <= 0:
            continue
        z = (sp[m] - mu) / sd
        if pos is None:
            if abs(z) >= zin and abs(sp[m] - mu) >= mb:
                e = m + 1
                aN, aS = n["atm"][e], s["atm"][e]
                if aN < 0 or aS < 0:
                    continue
                pos = dict(e=e, dir=-np.sign(z), aN=int(aN), aS=int(aS), z=z)   # dir +1 = long NIFTY
        else:
            done = (pos["dir"] < 0 and z <= zout) or (pos["dir"] > 0 and z >= -zout) or (m - pos["e"] >= hold)
            if done or m == T_END - 1:
                trades.append(close(n, s, pos, m + 1, slip))
                pos = None
    if pos is not None:
        trades.append(close(n, s, pos, T_END, slip))
    return [t for t in trades if t is not None]


def close(n, s, pos, x, slip):
    e = pos["e"]
    qN = n["lot"]
    FN, FS = n["fut"][e], s["fut"][e]
    if not (np.isfinite(FN) and np.isfinite(FS)):
        return None
    qS = qN * FN / FS                                    # notional match in SENSEX units
    gross = chg = 0.0
    for d, a, q, sgn in ((n, pos["aN"], qN, pos["dir"]), (s, pos["aS"], qS, -pos["dir"])):
        ce_e, pe_e = round(float(d["px"][0, a, e]), 2), round(float(d["px"][1, a, e]), 2)
        ce_x, pe_x = round(float(d["px"][0, a, x]), 2), round(float(d["px"][1, a, x]), 2)
        if np.isnan([ce_e, pe_e, ce_x, pe_x]).any():
            return None
        # long synthetic: buy CE, sell PE ; short synthetic: sell CE, buy PE
        if sgn > 0:
            legs = [(ce_x - slip, ce_e + slip), (pe_e - slip, pe_x + slip)]   # (sell, buy) per leg
        else:
            legs = [(ce_e - slip, ce_x + slip), (pe_x - slip, pe_e + slip)]
        for sell, buy in legs:
            sell, buy = max(sell, 0) * q, buy * q
            gross += sell - buy
            chg += leg_cost(d["index"], sell, buy)
    return dict(e=e, x=x, dir=pos["dir"], gross=gross, charges=chg, net=gross - chg)
