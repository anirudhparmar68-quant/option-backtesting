"""Variance-risk-premium harvest with defined risk and no stop-loss.

Entry minutes E (independent positions, 1 lot per leg): sell the ATM straddle (synthetic
ATM, nearest expiry), buy wings W strikes away on both sides. Held to the time exit
(15:15 pre-CAS / expiry days, else 15:35) and closed at that minute's price. No stop:
the wings cap the loss, so no intra-minute fill assumption is needed.
Costs: statutory charges + `slip` ticks per leg per fill.
"""
import numpy as np

import straddle_920 as st

STEP = {"NIFTY": 50.0, "SENSEX": 100.0}


def fly_pnl(d, E, W, slip_ticks=1):
    """Net P&L per entry in E (NaN where the fly cannot be built)."""
    K, px, atm, X, lot = d["strikes"], d["px"], d["atm"], d["X"], d["lot"]
    step = STEP[d["index"]]
    pos = {float(k): i for i, k in enumerate(K)}
    slip = slip_ticks * st.TICK
    out = np.full(len(E), np.nan)
    for n, m in enumerate(E):
        a = atm[m]
        if a < 0:
            continue
        ka = float(K[a])
        c, p = pos.get(ka + W * step), pos.get(ka - W * step)
        if c is None or p is None:
            continue
        legs = [(0, a, -1), (1, a, -1), (0, c, +1), (1, p, +1)]      # -1 short, +1 long
        e = np.array([np.round(float(px[j, s, m]), 2) for j, s, _ in legs])
        x = np.array([np.round(float(px[j, s, X]), 2) for j, s, _ in legs])
        if np.isnan(e).any() or np.isnan(x).any():
            continue
        gross = chg = 0.0
        for (j, s, side), ei, xi in zip(legs, e, x):
            if side < 0:
                sell, buy = max(ei - slip, 0) * lot, (xi + slip) * lot
            else:
                buy, sell = (ei + slip) * lot, max(xi - slip, 0) * lot
            gross += sell - buy
            chg += st.charges(d["index"], sell, buy)
        out[n] = gross - chg
    return out
