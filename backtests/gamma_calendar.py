"""Gamma-neutral calendar ("theta carry with a term-structure hedge").

At each entry minute:
  * K = strike nearest the near-expiry synthetic future that is listed in both expiries.
  * Short 1 lot of the NEAR ATM straddle, long r lots of the NEXT-expiry straddle at K,
    r = gamma(near) / gamma(next) (Black-76, each straddle's own implied vol), capped.
    Net gamma ~ 0 at entry; net theta > 0 (near decays faster); net vega > 0 (the long
    leg has more vega), so volatility spikes help rather than hurt.
  * Exit: time exit (15:15 pre-CAS / on expiry days, 15:35 otherwise) at that minute's
    price; optional portfolio stop judged on minute closes, executed at the next minute's
    price. No intra-minute high/low is used for any decision or fill.
Costs: statutory charges + `slip` ticks per leg per fill (scaled by quantity).
"""

import math

import numpy as np

import straddle_920 as st

YEAR = 365 * 86400


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def straddle_iv(price, F, K, T):
    lo, hi = 1e-4, 5.0
    for _ in range(60):
        v = (lo + hi) / 2
        d1 = (math.log(F / K) + 0.5 * v * v * T) / (v * math.sqrt(T))
        d2 = d1 - v * math.sqrt(T)
        val = F * ncdf(d1) - K * ncdf(d2) + K * ncdf(-d2) - F * ncdf(-d1)
        lo, hi = (lo, v) if val > price else (v, hi)
    return (lo + hi) / 2


def straddle_greeks(price, F, K, T):
    v = straddle_iv(price, F, K, T)
    d1 = (math.log(F / K) + 0.5 * v * v * T) / (v * math.sqrt(T))
    gamma = 2 * npdf(d1) / (F * v * math.sqrt(T))
    vega = 2 * F * npdf(d1) * math.sqrt(T)
    return v, gamma, vega


def t_years(day, expiry, m):
    import datetime as dt
    now = dt.datetime.fromisoformat(day + "T09:15:00") + dt.timedelta(minutes=int(m))
    exp = dt.datetime.fromisoformat(expiry + "T15:30:00")
    return (exp - now).total_seconds() / YEAR


def trade(d, m, p):
    """One calendar entered at minute m. Returns dict or None."""
    F = d["fut0"][m]
    if not np.isfinite(F):
        return None
    k0, k1 = d["k0"], d["k1"]
    common = np.intersect1d(k0, k1)
    if not len(common):
        return None
    K = common[np.argmin(np.abs(common - F))]
    s0, s1 = int(np.searchsorted(k0, K)), int(np.searchsorted(k1, K))
    e0 = d["px0"][:, s0, m].astype(float)
    e1 = d["px1"][:, s1, m].astype(float)
    if np.isnan(e0).any() or np.isnan(e1).any() or e0.sum() <= 0 or e1.sum() <= 0:
        return None
    T0, T1 = t_years(d["day"], str(d["exp0"]), m), t_years(d["day"], str(d["exp1"]), m)
    F1 = d["fut1"][m] if np.isfinite(d["fut1"][m]) else F
    try:
        iv0, g0, v0 = straddle_greeks(e0.sum(), F, K, T0)
        iv1, g1, v1 = straddle_greeks(e1.sum(), F1, K, T1)
    except (ValueError, ZeroDivisionError):
        return None
    r = min(g0 / g1, p.get("rmax", 4.0))
    X = d["X"]
    exit_m = X
    lot, slip = d["lot"], p.get("slip", 0) * st.TICK
    cl0, cl1 = d["cl0"][:, s0, :], d["cl1"][:, s1, :]
    stop = p.get("stop")                 # portfolio stop in points of near straddle credit
    if stop is not None:
        mtm = (e0.sum() - cl0.sum(0)) + r * (cl1.sum(0) - e1.sum())   # per near-lot unit
        thr = -stop * e0.sum()
        hit = np.flatnonzero((np.arange(st.N_MIN) >= m) & (np.arange(st.N_MIN) < X - 1) & (mtm <= thr))
        if len(hit):
            exit_m = int(hit[0]) + 1                                    # act on the next minute
    x0 = d["px0"][:, s0, exit_m].astype(float)
    x1 = d["px1"][:, s1, exit_m].astype(float)
    if np.isnan(x0).any() or np.isnan(x1).any():
        return None
    q0, q1 = lot, r * lot
    sell0, buy0 = (e0 - slip).clip(0) * q0, (x0 + slip) * q0          # short near
    buy1, sell1 = (e1 + slip) * q1, (x1 - slip).clip(0) * q1          # long next
    gross = float((sell0 - buy0).sum() + (sell1 - buy1).sum())
    chg = float(sum(st.charges(d["index"], sell0[j], buy0[j]) + st.charges(d["index"], sell1[j], buy1[j])
                    for j in (0, 1)))
    return dict(gross=gross, charges=chg, net=gross - chg, r=r, iv0=iv0, iv1=iv1,
                vega_net=(r * v1 - v0) * lot, exit=exit_m, K=float(K))


def run(days, p):
    """{(index, day): net}"""
    ent = [st.mi(t) for t in p["entries"]]
    out = {}
    for d in days:
        if p.get("skip_expiry") and str(d["exp0"]) == d["day"]:
            continue
        tot = 0.0
        for m in ent:
            t = trade(d, m, p)
            if t is not None:
                tot += t["net"]
        out[(d["index"], d["day"])] = tot
    return out
