"""Reference loop for the index-move stop mode (trig='index') of straddle_fast.py.
Run: python tests/test_index_stop.py"""
import os, random, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backtests"))
import straddle_920 as st
import straddle_fast as sf


def ref_entry(d, m0, p):
    X, lot, slip = d["X"], d["lot"], p["slip"] * st.TICK
    r2 = lambda x: round(float(x), 2)
    total, m, att = 0.0, m0, 0
    while True:
        a = d["atm"][m]
        S0 = d["spot_open"][m]
        if a < 0:
            return total
        e = [r2(d["px"][j, a, m]) for j in (0, 1)]
        if np.isnan(e).any() or not np.isfinite(S0):
            return total
        ex, cost = [None, None], [False, False]
        for t in range(m, X):
            c = d["spot_close"][t]
            hit = []
            for j in (0, 1):
                if ex[j] is not None:
                    continue
                if cost[j]:
                    oc = d["close"][j, a, t]
                    if not np.isnan(oc) and r2(oc) >= e[j] * p["cost"]:
                        hit.append(j)
                elif np.isfinite(c) and ((j == 0 and c >= S0 * (1 + p["mv"])) or (j == 1 and c <= S0 * (1 - p["mv"]))):
                    hit.append(j)
            for j in hit:
                ex[j] = t
            if len(hit) == 1 and ex[1 - hit[0]] is None:
                cost[1 - hit[0]] = True
            if ex[0] is not None and ex[1] is not None:
                break
        for j in (0, 1):
            f = r2(d["px"][j, a, min(ex[j] + 1, X)]) if ex[j] is not None else r2(d["px"][j, a, X])
            sell, buy = max(e[j] - slip, 0) * lot, (f + slip) * lot
            total += sell - buy - st.charges(d["index"], sell, buy)
        if ex[0] is None or ex[1] is None or att >= p["max_re"] or max(ex) + 1 > sf.RE_LAST:
            return total
        m, att = max(ex) + 1, att + 1


days = sf.attach_spot(sf.load())
random.seed(9)
worst = 0
for d in random.sample(days, 25):
    for p in (dict(sl=0.2, mv=0.002, cost=1.0, max_re=1, slip=0, trig="index"),
              dict(sl=0.2, mv=0.004, cost=1.0, max_re=1, slip=1, trig="index")):
        E = np.arange(st.mi("09:20"), st.mi("13:00") + 1)
        ref = sum(ref_entry(d, int(m), p) for m in E)
        fast = sf.entry_pnl(d, p).sum()
        worst = max(worst, abs(ref - fast))
        assert abs(ref - fast) < 1, (d["index"], d["day"], p, ref, fast)
print("ok, max |diff| =", round(worst, 4))
