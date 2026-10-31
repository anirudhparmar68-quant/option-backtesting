"""Genuine S2 - staggered NIFTY iron condor (same engine/costs as ironfly_s1.py).

Strike choice at each entry minute:
  * ATM by synthetic future; straddle = ATM CE + ATM PE price ("mid" = LTP here).
  * n = straddle rounded UP to whole strike steps; short CE = ATM + n steps,
    short PE = ATM - n steps.
  * Wing width w = 1..10 steps, same on both sides; take the w whose
    max loss / net credit = (w*step - credit) / credit is nearest RATIO (3);
    a tie goes to the narrower wing.
Stops, cost-to-cost, target, re-entry, time exit, fills, slippage and charges: exactly as S1.

Data caveat: data/upstox_7delta.db holds NIFTY strikes down to ~7 delta only, so far
wings may be missing. Only widths whose wing strikes have data are considered; a pick is
'bracketed' when an available width reaches the ratio (so the true nearest width is
available), else 'truncated'. pick_coverage() reports how often each happens.
"""

import math

import numpy as np

import ironfly_s1 as s

MAX_W = 10


def pick_condor(d, ratio=3.0):
    """Per minute: (sc, sp, cw, pw) strike indices (-1 if not possible) and a
    'bracketed' flag."""
    k, px, atm = d["strikes"], d["px"], d["atm"]
    N = len(atm)
    out = np.full((4, N), -1, np.int32)
    bracketed = np.zeros(N, bool)
    if len(k) < 2:
        return out, bracketed
    step = float(np.min(np.diff(k)))
    where = {float(x): i for i, x in enumerate(k)}

    def idx(strike):
        return where.get(float(round(strike / step) * step), -1)

    for m in range(N):
        a = atm[m]
        if a < 0:
            continue
        straddle = px[0, a, m] + px[1, a, m]
        if not np.isfinite(straddle) or straddle <= 0:
            continue
        n = math.ceil(straddle / step - 1e-9)
        sc, sp = idx(k[a] + n * step), idx(k[a] - n * step)
        if sc < 0 or sp < 0 or np.isnan(px[0, sc, m]) or np.isnan(px[1, sp, m]):
            continue
        best, best_err, reach = None, np.inf, False
        for w in range(1, MAX_W + 1):
            cw, pw = idx(k[sc] + w * step), idx(k[sp] - w * step)
            if cw < 0 or pw < 0 or np.isnan(px[0, cw, m]) or np.isnan(px[1, pw, m]):
                continue
            credit = (px[0, sc, m] - px[0, cw, m]) + (px[1, sp, m] - px[1, pw, m])
            if credit <= 0:
                continue
            r = (w * step - credit) / credit
            reach |= r >= ratio
            err = abs(r - ratio)
            if err < best_err - 1e-12:             # strict: tie keeps the narrower
                best, best_err = (cw, pw), err
        if best is None:
            continue
        out[:, m] = (sc, sp, best[0], best[1])
        bracketed[m] = reach
    return out, bracketed


def run_day(d, p, picks=None):
    legs, _ = picks if picks is not None else pick_condor(d, p.get("ratio", 3.0))
    E = np.arange(s.mi("09:20"), s.mi(p["entry_last"]) + 1)
    first = s.sim_legs(d, E, *[legs[i][E] for i in range(4)], p)
    if first is None:
        return []
    out = [("first", first)]
    if p.get("reentry", True) and first["reenter"].any():
        E2 = np.maximum(first["exit_c"], first["exit_p"])[first["reenter"]] + 1
        second = s.sim_legs(d, E2, *[legs[i][E2] for i in range(4)], p)
        if second is not None:
            out.append(("reentry", second))
    return out


DEFAULT = dict(s.DEFAULT, ratio=3.0)


def run_all(days, p=DEFAULT, picks=None):
    daily = {}
    for d in days:
        res = run_day(d, p, None if picks is None else picks[d["day"]])
        daily[d["day"]] = sum(float(r["net"].sum()) for _, r in res)
    return daily


if __name__ == "__main__":
    import time
    t0 = time.time()
    days = s.load_cache()
    picks = {d["day"]: pick_condor(d) for d in days}
    E = np.arange(s.mi("09:20"), s.mi("13:00") + 1)
    tot = sum(len(E) for _ in days)
    got = sum(int((picks[d["day"]][0][0][E] >= 0).sum()) for d in days)
    br = sum(int(picks[d["day"]][1][E].sum()) for d in days)
    print(f"entries {tot}: condor possible {got} ({got / tot:.1%}), bracketed {br}"
          f" ({br / max(got, 1):.1%} of possible)  [{time.time() - t0:.0f}s]")
