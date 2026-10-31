"""Genuine S1 - staggered NIFTY iron fly, on data/upstox_7delta.db (1-minute LTP candles).

Rules (each fly 1 lot per leg):
  1. Every minute 09:20 .. ENTRY_LAST (default 13:00): sell ATM CE + PE, buy one CE wing
     and one PE wing.
  2. ATM = strike where CE and PE prices are closest (synthetic future). Expiry: nearest;
     on NIFTY's own expiry day, the next one.
  3. Wings: OTM CE / PE whose delta is nearest WING_DELTA (default 0.10), only 5-15 delta
     allowed; a tie goes to the strike nearer the money.
  4. Side = short + its wing. Credit = short sell fill - wing buy fill.
     Value = short price - wing price (here: last traded price = minute close, ffilled).
  5. Side stop: value >= credit * (1 + SL) -> close that side.
  6. After one side stops, the other side's stop moves to its own credit (cost), from the
     next minute. Both hitting in the same minute -> both close as stops.
  7. Target: while both sides open, call value + put value <= total credit * (1 - TGT)
     -> close all four legs. Stops are checked first.
  8. Re-entry: both sides closed by stops AND fly lost money before charges -> one fresh
     fly at the next minute (new ATM/wings), only if the second side closed before 14:50.
     Never after a target or the time exit; the re-entry never re-enters.
  9. Time exit: 15:35 on closing-auction (CAS) days, 15:20 before CAS (detected per day:
     bars after 15:30 exist -> CAS day).

Data limits / modelling: no bid/ask in the data, so "mid" = 1-minute close (LTP). Entries
fill at the entry minute's open (else last close). Stops/targets are judged on minute
closes and fill at those closes. Time exit fills at the exit minute's open. Slippage:
SLIP_TICKS per leg per fill (default 3 x Rs 0.05). Costs: brokerage 0 + statutory charges.
Delta: Black-76 on the synthetic future with each option's own implied vol, time to
15:30 on expiry.
"""

import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np

IST = timezone(timedelta(hours=5, minutes=30))
DAY_START = 9 * 60 + 15
N_MIN = (15 * 60 + 40) - DAY_START           # 09:15 .. 15:39
TICK = 0.05
CACHE_DIR = "data/cache_s1"

STT_SELL, SEBI, STAMP_BUY, GST, NSE_TXN = 0.10 / 100, 10 / 1e7, 0.003 / 100, 0.18, 0.03503 / 100


def mi(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m - DAY_START


def hhmm(i):
    t = DAY_START + int(i)
    return f"{t // 60:02d}:{t % 60:02d}"


def charges(sell_value, buy_value):
    turnover = sell_value + buy_value
    txn, sebi = turnover * NSE_TXN, turnover * SEBI
    return sell_value * STT_SELL + buy_value * STAMP_BUY + txn + sebi + GST * (txn + sebi)


# ---------------------------------------------------------------- delta maths
def _ncdf(x):
    z = np.abs(x) / math.sqrt(2)
    t = 1 / (1 + 0.3275911 * z)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
              - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1 + np.sign(x) * y)


def black76_delta(price, fut, k, t, call):
    """Arrays broadcast together. Delta from each option's own IV; NaN if IV undefined."""
    intrinsic = np.maximum(fut - k, 0) if call else np.maximum(k - fut, 0)
    ok = ~np.isnan(price) & (price > intrinsic + 1e-9) & (price > 0) & ~np.isnan(fut)
    shape = np.broadcast(price, fut, k, t).shape
    lo, hi = np.full(shape, 1e-4), np.full(shape, 5.0)
    sq = np.sqrt(t)
    lf = np.log(fut / k)
    for _ in range(50):
        vol = (lo + hi) / 2
        d1 = (lf + 0.5 * vol ** 2 * t) / (vol * sq)
        d2 = d1 - vol * sq
        val = (fut * _ncdf(d1) - k * _ncdf(d2)) if call else (k * _ncdf(-d2) - fut * _ncdf(-d1))
        high = val > price
        hi = np.where(high, vol, hi)
        lo = np.where(high, lo, vol)
    vol = (lo + hi) / 2
    d1 = (lf + 0.5 * vol ** 2 * t) / (vol * sq)
    delta = _ncdf(d1) if call else _ncdf(d1) - 1
    return np.where(ok, delta, np.nan)


# ---------------------------------------------------------------- cache
def trading_plan(db):
    """[(day, expiry)] for NIFTY: nearest expiry, next one on expiry day."""
    exps = [r[0] for r in db.execute(
        "SELECT DISTINCT expiry FROM contracts WHERE underlying='NIFTY' AND n_candles > 0"
        " ORDER BY 1")]
    days = [r[0] for r in db.execute(
        "SELECT DISTINCT date(o.ts,'unixepoch','+330 minutes') FROM option_candles o"
        " JOIN contracts c ON c.id=o.contract_id WHERE c.underlying='NIFTY' ORDER BY 1")]
    plan = []
    for d in days:
        up = [e for e in exps if e >= d]
        if up and up[0] == d:
            up = up[1:]
        if up:
            plan.append((d, up[0]))
    return plan


def build_day(db, day, expiry):
    start = int(datetime.fromisoformat(day + "T09:15:00").replace(tzinfo=IST).timestamp())
    rows = db.execute(
        "SELECT c.strike, c.option_type, o.ts, o.open, o.close FROM option_candles o"
        " JOIN contracts c ON c.id = o.contract_id WHERE c.underlying='NIFTY' AND c.expiry=?"
        " AND o.ts >= ? AND o.ts < ?", (expiry, start, start + N_MIN * 60)).fetchall()
    if not rows:
        return None
    strikes = np.array(sorted({r[0] for r in rows}))
    pos = {k: i for i, k in enumerate(strikes)}
    opn = np.full((2, len(strikes), N_MIN), np.nan, np.float32)
    cls = np.full((2, len(strikes), N_MIN), np.nan, np.float32)
    last_m = 0
    for k, ot, ts, o, c in rows:
        j, m = (0 if ot == "CE" else 1), (ts - start) // 60
        opn[j, pos[k], m], cls[j, pos[k], m] = o / 100, c / 100
        last_m = max(last_m, m)
    ff = cls.copy()
    for m in range(1, N_MIN):
        gap = np.isnan(ff[:, :, m])
        ff[:, :, m][gap] = ff[:, :, m - 1][gap]
    # price at the start of minute m: open, else last close before it
    prev = np.concatenate([np.full((2, len(strikes), 1), np.nan, np.float32), ff[:, :, :-1]], 2)
    px = np.where(np.isnan(opn), prev, opn)

    # synthetic future / ATM per minute
    ce, pe = px[0], px[1]
    ok = ~np.isnan(ce) & ~np.isnan(pe)
    diff = np.where(ok, ce - pe, np.inf)
    s_min = np.argmin(np.abs(diff), axis=0)
    fut = strikes[s_min] + diff[s_min, np.arange(N_MIN)]
    fut[~ok.any(axis=0)] = np.nan
    dist = np.where(ok, np.abs(strikes[:, None] - fut[None, :]), np.inf)
    atm = np.argmin(dist, axis=0).astype(np.int16)
    atm[~ok.any(axis=0)] = -1

    exp_ts = datetime.fromisoformat(expiry + "T15:30:00").replace(tzinfo=IST).timestamp()
    t = (exp_ts - (start + np.arange(N_MIN) * 60)) / (365 * 86400)
    dce = black76_delta(px[0].astype(float), fut[None, :], strikes[:, None], t[None, :], True)
    dpe = black76_delta(px[1].astype(float), fut[None, :], strikes[:, None], t[None, :], False)
    lot = db.execute("SELECT lot_size FROM contracts WHERE underlying='NIFTY' AND expiry=? LIMIT 1",
                     (expiry,)).fetchone()[0]
    return dict(strikes=strikes, px=px, close=ff, fut=fut.astype(np.float32), atm=atm,
                dce=dce.astype(np.float32), dpe=dpe.astype(np.float32), lot=lot,
                cas=last_m >= mi("15:31"), first_m=min((r[2] - start) // 60 for r in rows))


def load_cache(db_path="data/upstox_7delta.db", rebuild=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = []
    for day, expiry in trading_plan(db):
        f = os.path.join(CACHE_DIR, f"{day}_{expiry}.npz")
        if rebuild or not os.path.exists(f):
            d = build_day(db, day, expiry)
            if d is None:
                continue
            np.savez(f, **d)
        z = np.load(f)
        d = {k: z[k] for k in z.files}
        d["lot"], d["cas"], d["first_m"] = int(d["lot"]), bool(d["cas"]), int(d["first_m"])
        if d["first_m"] > mi("09:20"):          # special sessions (e.g. Muhurat) -> skip
            continue
        d["day"], d["expiry"] = day, expiry
        out.append(d)
    return out


# ---------------------------------------------------------------- simulation
def pick_wings(d, wing_delta, lo=0.05, hi=0.15):
    """CE / PE wing strike index per minute (-1 if none in [lo, hi])."""
    k, fut, atm = d["strikes"], d["fut"], d["atm"]
    out = []
    for dl, call in ((d["dce"], True), (np.abs(d["dpe"]), False)):
        atm_k = np.where(atm >= 0, k[np.maximum(atm, 0)], np.nan)
        otm = (k[:, None] > atm_k[None, :]) if call else (k[:, None] < atm_k[None, :])
        valid = otm & (dl >= lo) & (dl <= hi)
        err = np.abs(dl - wing_delta) + 1e-9 * np.abs(k[:, None] - fut[None, :])
        err = np.where(valid, err, np.inf)
        w = np.argmin(err, axis=0)
        w[~valid.any(axis=0)] = -1
        out.append(w)
    return out


def first_true(mask):
    """First column index where mask is True per row, N_MIN if none."""
    any_ = mask.any(axis=1)
    return np.where(any_, mask.argmax(axis=1), N_MIN)


def sim_flies(d, E, wings, p):
    """Simulate flies entered at minutes E (array). Returns dict of per-fly arrays."""
    atm = d["atm"][E]
    return sim_legs(d, E, atm, atm, wings[0][E], wings[1][E], p)


def sim_legs(d, E, sc, sp, cw, pw, p):
    """Four-leg position per entry minute: short call sc / short put sp, wings cw / pw
    (strike indices; -1 = unavailable). Returns dict of per-position arrays."""
    ok = (sc >= 0) & (sp >= 0) & (cw >= 0) & (pw >= 0)
    E, sc, sp, cw, pw = E[ok], sc[ok], sp[ok], cw[ok], pw[ok]
    n = len(E)
    if n == 0:
        return None
    slip = p["slip_ticks"] * TICK
    px, cl, lot = d["px"], d["close"], d["lot"]
    X = mi("15:35") if d["cas"] else mi("15:20")

    s_ce, s_pe = px[0, sc, E] - slip, px[1, sp, E] - slip          # short sells
    b_cw, b_pw = px[0, cw, E] + slip, px[1, pw, E] + slip            # wing buys
    Cc, Cp = s_ce - b_cw, s_pe - b_pw
    good = (Cc > 0) & (Cp > 0) & ~np.isnan(Cc) & ~np.isnan(Cp)

    Vc = cl[0, sc, :] - cl[0, cw, :]                                 # (n, N_MIN)
    Vp = cl[1, sp, :] - cl[1, pw, :]
    grid = np.arange(N_MIN)[None, :]
    live = (grid >= E[:, None]) & (grid < X)
    sl = 1 + p["sl"]
    tc = first_true(live & (Vc >= sl * Cc[:, None]))
    tp = first_true(live & (Vp >= sl * Cp[:, None]))
    if p["tgt"] is not None:
        tt = first_true(live & (Vc + Vp <= (1 - p["tgt"]) * (Cc + Cp)[:, None]))
    else:
        tt = np.full(n, N_MIN)
    t1 = np.minimum(tc, tp)
    target = (tt < t1) & (tt < X)
    # second side after the first stop: stop at cost from the next minute
    after = live & (grid > t1[:, None])
    tc2 = first_true(after & (Vc >= Cc[:, None]))
    tp2 = first_true(after & (Vp >= Cp[:, None]))
    exit_c = np.where(target, tt, np.where(tc <= tp, tc, np.minimum(tc2, X)))
    exit_p = np.where(target, tt, np.where(tp <= tc, tp, np.minimum(tp2, X)))
    exit_c = np.where(t1 >= X, X, exit_c)
    exit_p = np.where(t1 >= X, X, exit_p)
    exit_c, exit_p = np.where(target, tt, exit_c), np.where(target, tt, exit_p)

    def exit_px(j, s, t):
        timed = t >= X
        tt_ = np.minimum(t, N_MIN - 1)
        return np.where(timed, px[j, s, X], cl[j, s, tt_])

    bb_ce = exit_px(0, sc, exit_c) + slip         # buy back short
    sw_ce = np.maximum(exit_px(0, cw, exit_c) - slip, 0)
    bb_pe = exit_px(1, sp, exit_p) + slip
    sw_pe = np.maximum(exit_px(1, pw, exit_p) - slip, 0)
    gross = ((s_ce - bb_ce) + (sw_ce - b_cw) + (s_pe - bb_pe) + (sw_pe - b_pw)) * lot
    chg = (charges(s_ce * lot, bb_ce * lot) + charges(sw_ce * lot, b_cw * lot)
           + charges(s_pe * lot, bb_pe * lot) + charges(sw_pe * lot, b_pw * lot))
    both_stopped = ~target & (exit_c < X) & (exit_p < X)
    second = np.maximum(exit_c, exit_p)
    reenter = both_stopped & (gross < 0) & (second < mi("14:50"))
    reason = np.where(target, "target", np.where(both_stopped, "stops",
                      np.where((exit_c < X) | (exit_p < X), "stop+time", "time")))
    r = dict(E=E, sc=d["strikes"][sc], sp=d["strikes"][sp], cw=d["strikes"][cw], pw=d["strikes"][pw],
             credit=(Cc + Cp), gross=gross, charges=chg, net=gross - chg,
             exit_c=exit_c, exit_p=exit_p, reason=reason, reenter=reenter & good, good=good)
    return {k: v[good] for k, v in r.items()}


def run_day(d, p, wings_cache=None):
    wings = (wings_cache or {}).get(p["wing"]) or pick_wings(d, p["wing"])
    E = np.arange(mi("09:20"), mi(p["entry_last"]) + 1)
    first = sim_flies(d, E, wings, p)
    if first is None:
        return []
    out = [("first", first)]
    if p.get("reentry", True) and first["reenter"].any():
        E2 = np.maximum(first["exit_c"], first["exit_p"])[first["reenter"]] + 1
        second = sim_flies(d, E2, wings, p)
        if second is not None:
            out.append(("reentry", second))
    return out


DEFAULT = dict(sl=0.20, tgt=0.20, wing=0.10, entry_last="13:00", slip_ticks=3, reentry=True)


def run_all(days, p=DEFAULT, wings_cache=None):
    """{day: net} plus per-fly records count."""
    daily, flies, reentries = {}, 0, 0
    for d in days:
        res = run_day(d, p, None if wings_cache is None else wings_cache.get(d["day"]))
        net = sum(float(r["net"].sum()) for _, r in res)
        daily[d["day"]] = net
        for kind, r in res:
            if kind == "first":
                flies += len(r["net"])
            else:
                reentries += len(r["net"])
    return daily, flies, reentries


if __name__ == "__main__":
    import time
    t0 = time.time()
    days = load_cache()
    print(f"{len(days)} days cached in {time.time() - t0:.0f}s")
    t0 = time.time()
    daily, flies, re = run_all(days)
    v = np.array(list(daily.values()))
    eq = np.cumsum(v)
    print(f"baseline: net {v.sum():,.0f}  flies {flies}  reentries {re}  win days"
          f" {(v > 0).sum()}/{len(v)}  maxDD {np.max(np.maximum.accumulate(eq) - eq):,.0f}"
          f"  ({time.time() - t0:.0f}s)")
