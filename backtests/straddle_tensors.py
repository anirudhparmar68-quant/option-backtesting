"""Per-entry net P&L tensors for the staggered straddle over a grid of trade rules.
Output: data/straddle_tensors.npz with P[config, day, entry] (float32), configs, days.
Entries = every minute 09:20..13:00 (221). Rules grid: SL x survivor rule x re-entries."""
import itertools
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import straddle_fast as sf

SL = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75]
COST = [None, 0.9, 1.0, 1.1]
MAX_RE = [0, 1, 2, 3]
CONFIGS = list(itertools.product(SL, COST, MAX_RE))
DAYS = None


def _init():
    global DAYS
    DAYS = sf.load(parallel=False)


def _one(i):
    out = np.zeros((len(CONFIGS), 221), np.float32)
    d = DAYS[i]
    for c, (sl, cost, re) in enumerate(CONFIGS):
        out[c] = sf.entry_pnl(d, dict(sl=sl, cost=cost, max_re=re))
    return i, out


if __name__ == "__main__":
    days = sf.load()
    P = np.zeros((len(CONFIGS), len(days), 221), np.float32)
    t0 = time.time()
    with Pool(4, initializer=_init) as pool:
        for k, (i, out) in enumerate(pool.imap_unordered(_one, range(len(days)), chunksize=2)):
            P[:, i] = out
            if k % 100 == 0:
                print(f"  {k}/{len(days)} days  {time.time() - t0:.0f}s", flush=True)
    np.savez("data/straddle_tensors.npz", P=P, configs=np.array(CONFIGS, dtype=object),
             index=np.array([d["index"] for d in days]), day=np.array([d["day"] for d in days]),
             expiry_day=np.array([d["expiry_day"] for d in days]), dte=np.array([d["dte"] for d in days]),
             fut920=np.array([d["fut"][5] for d in days]),
             futlast=np.array([d["fut"][~np.isnan(d["fut"])][-1] if (~np.isnan(d["fut"])).any() else np.nan
                               for d in days]))
    print("done", time.time() - t0)
