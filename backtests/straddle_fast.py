"""Vectorised version of the staggered short straddle (straddle_plain.py) for large
parameter studies, on data/upstox_7delta.db (NIFTY + SENSEX).

Per entry minute E (independent straddle, 1 lot per leg):
  * ATM by synthetic future at E; sell CE + PE at E's price (open, else last close).
  * Leg SL = entry * (1 + sl). Hit when the minute HIGH >= SL; fill at the SL, or at the
    minute open if it opens above (not on the entry minute) - same as straddle_920 'sl'.
    fill='close': fill at max(that, the trigger minute's close) - straddle_920 'close'.
    trig='index': leg stop on an underlying move of mv (CE up / PE down, index 1-min close),
    executed next minute; the survivor's cost stop is close-confirmed like trig='close'.
    Needs d['spot_open'] / d['spot_close'] (attach_spot()).
    trig='close': a stop triggers only when a minute CLOSES at/over the level and is
    executed at the next minute's price (open, else last close) - no intra-minute
    assumption; a re-entry opens at that same next minute.
  * cost: after one leg stops, the other leg's SL becomes cost_mult * its entry
    (1.0 = cost to cost; None = keep its original SL), from the next minute.
  * Both legs stopped -> re-entry at the next minute (new ATM), up to max_re times,
    only while that minute <= 14:50.
  * Time exit at the day's exit minute (15:15 pre-CAS and on expiry days after CAS,
    else 15:35), at that minute's price.
Costs: statutory charges (straddle_920.charges); optional slippage ticks per fill.
"""

import os
import sqlite3
from multiprocessing import Pool

import numpy as np

import straddle_920 as st

CACHE = "data/cache_straddle"
N = st.N_MIN
RE_LAST = st.mi("14:50")


def _build(job):
    db_path, index, day, expiry, nx = job          # nx: expiry day traded in the next expiry
    f = os.path.join(CACHE, f"{index}_{day}{'_nx' if nx else ''}.npz")
    if os.path.exists(f) and "strikes" in np.load(f).files:
        return f
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    b = st.DayBook(db, index, expiry, day)
    if not len(b.strikes) or np.isnan(b.open[:, :, st.mi("09:20")]).all():
        return None
    prev = np.concatenate([np.full(b.ffill.shape[:2] + (1,), np.nan), b.ffill[:, :, :-1]], 2)
    px = np.where(np.isnan(b.open), prev, b.open)
    atm = np.full(N, -1, np.int16)
    fut = np.full(N, np.nan)
    for m in range(st.mi("09:20"), st.mi("15:35") + 1):
        s, F = b.atm(m)
        if s is not None:
            atm[m], fut[m] = s, F
    cas = not np.isnan(b.high[:, :, st.mi("15:31"):]).all()
    x = st.mi("15:35") if (cas and expiry != day and not nx) else st.mi("15:15")
    np.savez(f, px=px.astype(np.float32), open=b.open.astype(np.float32),
             high=b.high.astype(np.float32), close=b.close.astype(np.float32), atm=atm, fut=fut,
             strikes=b.strikes,
             lot=b.lot, X=x,
             expiry_day=expiry == day, dte=(np.datetime64(expiry) - np.datetime64(day)).astype(int))
    return f


def load(db_path="data/upstox_7delta.db", index=None, parallel=True, next_on_expiry=False):
    """next_on_expiry: on an index's own expiry day use the NEXT expiry (exit still 15:15)."""
    os.makedirs(CACHE, exist_ok=True)
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    jobs = []
    for ix in ([index] if index else ["NIFTY", "SENSEX"]):
        exps = [r[0] for r in db.execute("SELECT DISTINCT expiry FROM contracts WHERE underlying=?"
                                         " AND n_candles > 0 ORDER BY 1", (ix,))]
        for (day,) in db.execute("SELECT DISTINCT date(o.ts,'unixepoch','+330 minutes') FROM"
                                 " option_candles o JOIN contracts c ON c.id=o.contract_id"
                                 " WHERE c.underlying=? ORDER BY 1", (ix,)):
            up = [e for e in exps if e >= day]
            nx = next_on_expiry and bool(up) and up[0] == day
            if nx:
                up = up[1:]
            if up:
                jobs.append((db_path, ix, day, up[0], nx))
    if parallel:
        with Pool(4) as pool:
            files = pool.map(_build, jobs, chunksize=4)
    else:                                   # e.g. inside a pool worker (no nested pools)
        files = [_build(j) for j in jobs]
    days = []
    for (_, ix, day, expiry, nx), f in zip(jobs, files):
        if f is None:
            continue
        z = np.load(f)
        d = {k: z[k] for k in z.files}
        d.update(index=ix, day=day, expiry=expiry, lot=int(d["lot"]), X=int(d["X"]),
                 expiry_day=bool(d["expiry_day"]), dte=int(d["dte"]))
        days.append(d)
    return days


def attach_spot(days):
    """Add the index's 1-minute open/close (data/{nifty,sensex}_index_1min.npz) to each day."""
    sp = {}
    for ix in ("NIFTY", "SENSEX"):
        z = np.load(f"data/{ix.lower()}_index_1min.npz")
        sp[ix] = {d: oc for d, oc in zip(z["days"], z["oc"])}
    keep = []
    for d in days:
        oc = sp[d["index"]].get(d["day"])
        if oc is None:
            continue
        d["spot_open"], d["spot_close"] = oc[:, 0].astype(float), oc[:, 1].astype(float)
        keep.append(d)
    return keep


def attach_premium_picks(days, target=100.0):
    """Per minute, the CE and the PE strike whose price is nearest `target` (-1 if none)."""
    for d in days:
        px = d["px"]
        for j, key in ((0, "pick_ce"), (1, "pick_pe")):
            dist = np.abs(px[j].astype(np.float64) - target)          # (S, N)
            dist = np.where(np.isnan(dist), np.inf, dist)
            idx = dist.argmin(0)
            idx[~np.isfinite(dist.min(0))] = -1
            d[key] = idx
    return days


def _first(mask):
    return np.where(mask.any(1), mask.argmax(1), N)


def legs(d, E, p):
    """Simulate straddles entered at minutes E. Returns per-straddle gross, charges,
    both_stopped flag, minute after the last stop."""
    if p.get("pick") == "premium":               # CE / PE strikes nearest a target premium
        a0, a1 = d["pick_ce"][E], d["pick_pe"][E]
    else:
        a0 = a1 = d["atm"][E]
    ok = (a0 >= 0) & (a1 >= 0)
    src = np.flatnonzero(ok)                     # positions in the caller's E
    E, a0, a1 = E[ok], a0[ok].astype(int), a1[ok].astype(int)
    A = (a0, a1)
    if not len(E):
        return None
    px, hi, op, X = d["px"], d["high"], d["open"], d["X"]
    slip = p.get("slip", 0) * st.TICK
    # float64 throughout: float32 turns e.g. a 60.00 SL into 60.000004 and misses touches
    # cache is float32: round back to exact paise so e.g. a 162.90 high still touches a
    # 162.90 SL
    e = np.round(np.stack([px[0, a0, E], px[1, a1, E]]).astype(np.float64), 2)
    good = ~np.isnan(e).any(0)
    grid = np.arange(N)[None, :]
    live = (grid >= E[:, None]) & (grid < X)
    H = np.round(np.stack([hi[0, a0, :], hi[1, a1, :]]).astype(np.float64), 2)   # (2, n, N)
    O = np.round(np.stack([op[0, a0, :], op[1, a1, :]]).astype(np.float64), 2)
    sl = e * (1 + p["sl"])
    close_basis = p.get("trig") in ("close", "index")   # judge on minute closes, act next minute
    if close_basis:
        H = np.round(np.stack([d["close"][0, a0, :], d["close"][1, a1, :]]).astype(np.float64), 2)
    if p.get("trig") == "index":
        # leg stop = underlying index move of p["mv"] from its level at entry (minute open):
        # CE stops when the index CLOSES >= S0*(1+mv), PE when it closes <= S0*(1-mv)
        sc_, so_ = d["spot_close"], d["spot_open"]
        S0 = so_[E]
        up = sc_[None, :] >= (S0 * (1 + p["mv"]))[:, None]
        dn = sc_[None, :] <= (S0 * (1 - p["mv"]))[:, None]
        t = np.stack([_first(live & up), _first(live & dn)])
        t = np.where(np.isfinite(S0)[None, :], t, N)
    else:
        t = np.stack([_first(live & (H[j] >= sl[j][:, None])) for j in (0, 1)])
    t1 = t.min(0)
    cm = p.get("cost")
    exit_t = t.copy()
    stop_px = np.where(t < N, sl, np.nan)
    if cm is not None:
        after = live & (grid > t1[:, None])
        for j in (0, 1):
            other_first = t[1 - j] < t[j]                          # this leg is the survivor
            lvl = e[j] * cm
            t2 = _first(after & (H[j] >= lvl[:, None]))
            if p.get("trig") == "index":
                # the survivor's index stop is replaced by the cost stop
                use = other_first
            else:
                use = other_first & (t2 <= t[j]) & (t2 < N)
            exit_t[j] = np.where(use, t2, t[j])
            stop_px[j] = np.where(use, lvl, stop_px[j])
    n = len(E)
    fill = np.empty((2, n))
    for j in (0, 1):
        tj = exit_t[j]
        stopped = tj < X
        tt = np.minimum(tj, N - 1)
        o = O[j][np.arange(n), tt]
        gap = stopped & (tt > E) & ~np.isnan(o) & (o > stop_px[j])
        f_stop = np.where(gap, o, stop_px[j])
        if p.get("fill", "sl") == "close":          # SL-M lands where the minute closed
            c = np.round(d["close"][j, A[j], tt].astype(np.float64), 2)
            f_stop = np.where(np.isnan(c), f_stop, np.maximum(f_stop, c))
        f_time = np.round(px[j, A[j], X].astype(np.float64), 2)
        if close_basis:                             # exit at the next minute's price
            f_stop = np.round(px[j, A[j], np.minimum(tt + 1, X)].astype(np.float64), 2)
        fill[j] = np.where(stopped, f_stop, f_time)
    lot = d["lot"]
    sell = np.maximum(e - slip, 0) * lot
    buy = (fill + slip) * lot
    gross = (sell - buy).sum(0)
    chg = sum(st.charges(d["index"], sell[j], buy[j]) for j in (0, 1))
    both = (exit_t < X).all(0)
    nxt = exit_t.max(0) + 1                           # close-basis: the exit minute itself
    return dict(E=E, gross=np.where(good, gross, 0.0), charges=np.where(good, chg, 0.0),
                both=both & good, nxt=nxt, src=src)


def entry_pnl(d, p, entries=None):
    """Net P&L per scheduled entry minute (incl. its re-entries), array over entries."""
    E0 = np.arange(st.mi("09:20"), st.mi("13:00") + 1) if entries is None else entries
    out = np.zeros(len(E0))
    pos = {int(m): i for i, m in enumerate(E0)}
    r = legs(d, E0, p)
    if r is None:
        return out
    owner = np.array([pos[int(m)] for m in r["E"]])     # E0 has no duplicates
    np.add.at(out, owner, r["gross"] - r["charges"])
    for _ in range(p.get("max_re", 1)):
        go = r["both"] & (r["nxt"] <= RE_LAST)
        if not go.any():
            break
        E2, own2 = r["nxt"][go], owner[go]
        r2 = legs(d, E2, p)
        if r2 is None:
            break
        owner2 = own2[r2["src"]]                 # legs() may drop entries without an ATM
        np.add.at(out, owner2, r2["gross"] - r2["charges"])
        r, owner = r2, owner2
    return out
