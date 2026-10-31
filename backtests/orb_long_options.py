"""Opening-range breakout with long options (no premium selling).

Per index-day (nearest expiry):
  * Opening range = high/low of the synthetic future over 09:15-OR_END.
  * After OR_END, the first minute the synthetic future is above the OR high (below the
    OR low) -> at the next minute buy the ATM CE (PE). One trade per index per day.
  * Exit when the synthetic future is back inside the range past its mid-point (judged
    at the start of a minute, executed at the next minute), else at 15:15.
Costs: statutory charges + `slip` ticks per fill. All fills at next-minute option prices.
"""
import numpy as np

import straddle_920 as st


def trade(d, or_end="09:45", slip=1, buffer_bps=0.0, last_entry="13:00", exit_at="15:15"):
    f = d["fut"]
    a0, a1 = st.mi("09:20"), st.mi(or_end)
    w = f[a0:a1 + 1]
    if not np.isfinite(w).all():
        return None
    hi, lo = w.max() * (1 + buffer_bps / 1e4), w.min() * (1 - buffer_bps / 1e4)
    mid = (w.max() + w.min()) / 2
    X = min(st.mi(exit_at), d["X"])
    for m in range(a1 + 1, st.mi(last_entry) + 1):
        if not np.isfinite(f[m]):
            continue
        side = 0 if f[m] > hi else 1 if f[m] < lo else None
        if side is None:
            continue
        e = m + 1
        a = d["atm"][e]
        if a < 0:
            return None
        pe = round(float(d["px"][side, a, e]), 2)
        x = X
        for t in range(e, X):
            if np.isfinite(f[t]) and ((side == 0 and f[t] < mid) or (side == 1 and f[t] > mid)):
                x = t + 1
                break
        px = round(float(d["px"][side, a, x]), 2)
        if np.isnan(pe) or np.isnan(px):
            return None
        s = slip * st.TICK
        buy, sell = (pe + s) * d["lot"], max(px - s, 0) * d["lot"]
        return dict(side="CE" if side == 0 else "PE", entry=st.hhmm(e), exit=st.hhmm(x),
                    net=sell - buy - st.charges(d["index"], sell, buy))
    return None
