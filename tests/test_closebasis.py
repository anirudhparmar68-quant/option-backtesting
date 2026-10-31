"""Reference loop for the close-basis / next-minute-execution straddle mode of
straddle_fast.py. Run: python tests/test_closebasis.py"""
import os, random, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backtests"))
import straddle_920 as st
import straddle_fast as sf


def ref_entry(d, m0, p):
    X, lot = d["X"], d["lot"]
    slip = p["slip"] * st.TICK
    r2 = lambda x: round(float(x), 2)
    total, m, att = 0.0, m0, 0
    while True:
        a = d["atm"][m]
        if a < 0:
            return total
        e = [r2(d["px"][j, a, m]) for j in (0, 1)]
        if any(np.isnan(e)):
            return total
        lvl = [x * (1 + p["sl"]) for x in e]
        ex = [None, None]
        for t in range(m, X):
            hit = [j for j in (0, 1) if ex[j] is None and not np.isnan(d["close"][j, a, t])
                   and r2(d["close"][j, a, t]) >= lvl[j]]
            for j in hit:
                ex[j] = t
            if len(hit) == 1 and ex[1 - hit[0]] is None:
                lvl[1 - hit[0]] = e[1 - hit[0]] * p["cost"]
            if ex[0] is not None and ex[1] is not None:
                break
        g = 0.0
        for j in (0, 1):
            f = r2(d["px"][j, a, min(ex[j] + 1, X)]) if ex[j] is not None else r2(d["px"][j, a, X])
            sell, buy = max(e[j] - slip, 0) * lot, (f + slip) * lot
            g += sell - buy - st.charges(d["index"], sell, buy)
        total += g
        if ex[0] is None or ex[1] is None or att >= p["max_re"]:
            return total
        nxt = max(ex) + 1
        if nxt > sf.RE_LAST:
            return total
        m, att = nxt, att + 1


days = sf.load()
random.seed(5)
worst = 0
for d in random.sample(days, 25):
    for p in (dict(sl=0.2, cost=1.0, max_re=1, slip=1, trig="close"), dict(sl=0.3, cost=1.0, max_re=3, slip=0, trig="close")):
        E = np.arange(st.mi("09:20"), st.mi("13:00") + 1)
        ref = sum(ref_entry(d, int(m), p) for m in E)
        fast = sf.entry_pnl(d, p).sum()
        worst = max(worst, abs(ref - fast))
        assert abs(ref - fast) < 1, (d["index"], d["day"], p, ref, fast)
print("ok, max |diff| =", round(worst, 4))
