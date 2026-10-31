"""S1 with an account-level profit lock that repeats through the day.

  * A "cycle" = all flies (and their re-entries) entered since the last square-off.
  * Cycle MTM each minute = what those flies would book if squared off at that minute's
    close (closed sides at their real exits, open legs at close +/- slippage), before charges.
  * When cycle MTM >= LOCK, every open fly of the cycle is squared off at that close and
    pending re-entries of the cycle are dropped. Scheduled entries resume from the next
    minute as a fresh cycle (until ENTRY_LAST). Unlimited cycles per day.
  * LOCK is given for a 65-qty lot and scaled by lot/65 on days with another lot size
    (25 in late 2024, 75 in 2025), so a lock means the same in points across the data.

Everything else is S1 (backtests/ironfly_s1.py).
"""

import numpy as np

import ironfly_s1 as s


def close_value(d, r, p):
    """(n, N_MIN) gross P&L if each fly were squared off at that minute's close;
    plus per-leg exit prices at that minute (4, n, N_MIN)."""
    slip = p["slip_ticks"] * s.TICK
    cl, lot = d["close"], d["lot"]
    sc, sp, cw, pw = r["idx"]
    s_ce, b_cw, s_pe, b_pw = r["fills"]
    grid = np.arange(s.N_MIN)[None, :]
    ex = np.empty((4,) + (len(sc), s.N_MIN))
    for k, (j, ix, sign, side_exit) in enumerate(((0, sc, +1, r["exit_c"]), (0, cw, -1, r["exit_c"]),
                                                   (1, sp, +1, r["exit_p"]), (1, pw, -1, r["exit_p"]))):
        live = cl[j, ix, :] + sign * slip
        if sign < 0:
            live = np.maximum(live, 0)
        closed = grid >= side_exit[:, None]
        ex[k] = np.where(closed, r["exits"][k][:, None], live)
    pnl = ((s_ce[:, None] - ex[0]) + (ex[1] - b_cw[:, None])
           + (s_pe[:, None] - ex[2]) + (ex[3] - b_pw[:, None])) * lot
    return pnl, ex


def run_day(d, p, lock, wings=None):
    """Returns (net, n_locks, lock_minutes)."""
    wings = wings or s.pick_wings(d, p["wing"])
    E = np.arange(s.mi("09:20"), s.mi(p["entry_last"]) + 1)
    first = s.sim_flies(d, E, wings, p)
    if first is None:
        return 0.0, 0, []
    groups = [first]
    parent = None
    if p.get("reentry", True) and first["reenter"].any():
        E2 = np.maximum(first["exit_c"], first["exit_p"])[first["reenter"]] + 1
        second = s.sim_flies(d, E2, wings, p)
        if second is not None:
            parent = first["E"][first["reenter"]][second["src"]]   # parent's entry minute
            groups.append(second)
    vals = [close_value(d, g, p) for g in groups]
    pnl = np.concatenate([v[0] for v in vals])
    ex = np.concatenate([v[1] for v in vals], axis=1)
    Ef = np.concatenate([g["E"] for g in groups])
    par = np.concatenate([groups[0]["E"]] + ([parent] if parent is not None else []))
    fills = np.concatenate([g["fills"] for g in groups], axis=1)
    final_t = np.concatenate([np.maximum(g["exit_c"], g["exit_p"]) for g in groups])
    X = int(groups[0]["X"][0])
    lot = d["lot"]
    lock_r = None if lock is None else lock * lot / 65

    killed_at = np.full(len(Ef), -1)
    alive = np.ones(len(Ef), bool)
    last_lock = -1
    locks = []
    if lock_r is not None:
        for m in range(s.mi("09:20"), X):
            # cycle members: parent entered after the last lock, fly already entered
            mem = alive & (par > last_lock) & (Ef <= m)
            if not mem.any():
                continue
            if pnl[mem, m].sum() >= lock_r:
                killed_at[mem] = m
                alive &= ~((par > last_lock) & (par <= m))   # drop members + pending re-entries
                locks.append(m)
                last_lock = m
    kept = (killed_at >= 0) | alive
    idx = np.arange(len(Ef))
    t = np.where(killed_at >= 0, killed_at, np.minimum(final_t, X))
    gross = np.where(killed_at >= 0, pnl[idx, np.minimum(t, s.N_MIN - 1)],
                     np.concatenate([g["gross"] for g in groups]))
    legs_exit = ex[:, idx, np.minimum(t, s.N_MIN - 1)]
    legs_exit = np.where(killed_at >= 0, legs_exit, np.concatenate([g["exits"] for g in groups], axis=1))
    s_ce, b_cw, s_pe, b_pw = fills
    bb_ce, sw_ce, bb_pe, sw_pe = legs_exit
    chg = (s.charges(s_ce * lot, bb_ce * lot) + s.charges(sw_ce * lot, b_cw * lot)
           + s.charges(s_pe * lot, bb_pe * lot) + s.charges(sw_pe * lot, b_pw * lot))
    net = float((gross - chg)[kept].sum())
    return net, len(locks), locks


def run_all(days, p, lock, wings_cache=None):
    daily, nl = {}, 0
    for d in days:
        w = None if wings_cache is None else wings_cache[d["day"]][p["wing"]]
        net, k, _ = run_day(d, p, lock, w)
        daily[d["day"]] = net
        nl += k
    return daily, nl
