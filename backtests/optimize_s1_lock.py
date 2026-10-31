"""Lock-level study for S1 + account profit lock (backtests/s1_lock.py).
Each lock level is run for the user's rules and for wing 7-delta + no re-entry, at 1 and
3 ticks of slippage; the lock level is then chosen walk-forward (quarterly, anchored,
neighbour-smoothed Calmar) so the OOS number is not a hindsight pick."""

import collections
import json
import os
import sys

import numpy as np

import ironfly_s1 as s
import s1_lock as L
from optimize_s1 import TEST_QUARTERS, stats

LOCKS = [None, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000, 300_000]
VARIANTS = {"rules": {}, "wing7_noRe": dict(wing=0.07, reentry=False)}

STOP_DAY = "--stop-day" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = args[0] if args else ("results/s1_daylock" if STOP_DAY else "results/s1_lock")
os.makedirs(out, exist_ok=True)
days = s.load_cache()
names = np.array([d["day"] for d in days])
wc = {d["day"]: {w: s.pick_wings(d, w) for w in (0.07, 0.10)} for d in days}
report = {}
for vname, extra in VARIANTS.items():
    for slip in (1, 3):
        p = dict(s.DEFAULT, slip_ticks=slip, **extra)
        mat, nlocks = [], []
        for lk in LOCKS:
            daily, nl = L.run_all(days, p, lk, wc, STOP_DAY)
            mat.append([daily[n] for n in names])
            nlocks.append(nl)
        mat = np.array(mat)
        rows = []
        for lk, v, nl in zip(LOCKS, mat, nlocks):
            st = stats(v)
            mon = collections.defaultdict(float)
            for n, x in zip(names, v):
                mon[n[:7]] += x
            oos = stats(v[names >= "2025-07-01"])
            rows.append(dict(lock=lk, locks_per_day=nl / len(days), loss_months=sum(x < 0 for x in mon.values()),
                             oos_net=oos["net"], oos_dd=oos["dd"], **st))
        # walk-forward choice of the lock level
        wf, chosen = [], []
        for lo, hi in TEST_QUARTERS:
            tr, te = names < lo, (names >= lo) & (names <= hi)
            cal = np.array([stats(v[tr])["calmar"] for v in mat])
            sm = np.array([cal[max(i - 1, 0):i + 2].mean() for i in range(len(cal))])
            b = int(np.argmax(sm))
            chosen.append(LOCKS[b])
            wf += list(mat[b, te])
        key = f"{vname}_slip{slip}"
        report[key] = dict(levels=rows, wf_chosen=chosen, wf_oos=stats(wf))
        print(f"\n== {key}  (walk-forward picks {chosen}: OOS net {stats(wf)['net']:,.0f}"
              f" dd {stats(wf)['dd']:,.0f})")
        for r in rows:
            print(f"  lock {str(r['lock']):>7}  locks/day {r['locks_per_day']:.2f}  net {r['net']:>12,.0f}"
                  f"  dd {r['dd']:>11,.0f}  calmar {r['calmar']:.2f}  win {r['win']:.0%}  lossM {r['loss_months']}"
                  f"  | Jul25+ net {r['oos_net']:>11,.0f} dd {r['oos_dd']:>10,.0f}")
with open(os.path.join(out, "report.json"), "w") as f:
    json.dump(report, f, indent=1, default=str)
