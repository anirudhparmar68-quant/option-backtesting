"""Per index-day cache with BOTH the near and the next expiry (data/upstox_7delta.db),
for term-structure strategies.

Saved per day (data/cache_calendar/{index}_{day}.npz), for k in (0 = near, 1 = next):
  px{k}    (2, S, N) price at the start of each minute (open, else last close)
  cl{k}    (2, S, N) last close (forward filled)
  k{k}     strikes; fut{k} synthetic future per minute; exp{k} expiry date
plus lot, X (time-exit minute: 15:15 pre-CAS and on expiry days, else 15:35).
"""

import os
import sqlite3
from multiprocessing import Pool

import numpy as np

import straddle_920 as st

CACHE = "data/cache_calendar"
N = st.N_MIN


def _book(db, index, expiry, day):
    b = st.DayBook(db, index, expiry, day)
    if not len(b.strikes):
        return None
    prev = np.concatenate([np.full(b.ffill.shape[:2] + (1,), np.nan), b.ffill[:, :, :-1]], 2)
    px = np.where(np.isnan(b.open), prev, b.open)
    fut = np.full(N, np.nan)
    for m in range(st.mi("09:20"), st.mi("15:35") + 1):
        s, F = b.atm(m)
        if s is not None:
            fut[m] = F
    return b, px, fut


def _build(job):
    db_path, index, day, e0, e1 = job
    f = os.path.join(CACHE, f"{index}_{day}.npz")
    if os.path.exists(f):
        return f
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = {}
    for k, e in ((0, e0), (1, e1)):
        r = _book(db, index, e, day)
        if r is None:
            return None
        b, px, fut = r
        out.update({f"px{k}": px.astype(np.float32), f"cl{k}": b.ffill.astype(np.float32),
                    f"k{k}": b.strikes, f"fut{k}": fut, f"exp{k}": e})
        if k == 0:
            cas = not np.isnan(b.high[:, :, st.mi("15:31"):]).all()
            out["lot"] = b.lot
            out["X"] = st.mi("15:35") if (cas and e != day) else st.mi("15:15")
    np.savez(f, **out)
    return f


def load(db_path="data/upstox_7delta.db"):
    os.makedirs(CACHE, exist_ok=True)
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    jobs = []
    for ix in ("NIFTY", "SENSEX"):
        exps = [r[0] for r in db.execute("SELECT DISTINCT expiry FROM contracts WHERE underlying=?"
                                         " AND n_candles > 0 ORDER BY 1", (ix,))]
        for (day,) in db.execute("SELECT DISTINCT date(o.ts,'unixepoch','+330 minutes') FROM"
                                 " option_candles o JOIN contracts c ON c.id=o.contract_id"
                                 " WHERE c.underlying=? ORDER BY 1", (ix,)):
            up = [e for e in exps if e >= day]
            if len(up) >= 2:
                jobs.append((db_path, ix, day, up[0], up[1]))
    with Pool(4) as pool:
        files = pool.map(_build, jobs, chunksize=4)
    days = []
    for (_, ix, day, e0, e1), f in zip(jobs, files):
        if f is None:
            continue
        z = np.load(f)
        d = {k: z[k] for k in z.files}
        d.update(index=ix, day=day, lot=int(d["lot"]), X=int(d["X"]), exp0=e0, exp1=e1)
        days.append(d)
    return days


if __name__ == "__main__":
    import time
    t0 = time.time()
    print(len(load()), "days", round(time.time() - t0), "s")
