"""Walk-forward parameter study for Genuine S2 (backtests/ironcondor_s2.py).
Same anti-overfitting design as optimize_s1.py (pre-declared grid, anchored quarterly
walk-forward, neighbour-smoothed Calmar). Usage: python backtests/optimize_s2.py"""

import argparse
import itertools
import json
import os
import time

import numpy as np

import ironcondor_s2 as c2
import ironfly_s1 as s
from optimize_s1 import TEST_QUARTERS, neighbours, stats

GRID = {
    "sl": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    "tgt": [None, 0.10, 0.20, 0.30],
    "ratio": [2.0, 3.0, 4.0],
    "entry_last": ["10:00", "11:00", "13:00"],
    "reentry": [True, False],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/s2_opt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    days = s.load_cache()
    names = np.array([d["day"] for d in days])
    picks = {r: {d["day"]: c2.pick_condor(d, r) for d in days} for r in GRID["ratio"]}

    keys = list(GRID)
    shape = tuple(len(GRID[k]) for k in keys)
    combos = list(itertools.product(*[range(n) for n in shape]))
    pos = {c: i for i, c in enumerate(combos)}
    mat = np.zeros((len(combos), len(days)))
    t0 = time.time()
    for ci, idx in enumerate(combos):
        p = dict(c2.DEFAULT, **{k: GRID[k][i] for k, i in zip(keys, idx)})
        daily = c2.run_all(days, p, picks[p["ratio"]])
        mat[ci] = [daily[n] for n in names]
        if ci % 50 == 0:
            print(f"  {ci}/{len(combos)} combos  {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(a.out, "daily_matrix.npy"), mat)

    def label(idx):
        return {k: GRID[k][i] for k, i in zip(keys, idx)}

    base = pos[tuple(GRID[k].index(c2.DEFAULT[k]) for k in keys)]
    folds, oos_sel, oos_base, oos_days = [], [], [], []
    for lo, hi in TEST_QUARTERS:
        train, test = names < lo, (names >= lo) & (names <= hi)
        cal = np.array([stats(mat[i, train])["calmar"] for i in range(len(combos))])
        cal = np.where(np.isfinite(cal), cal, np.nanmax(cal[np.isfinite(cal)]))
        smooth = np.array([np.mean([cal[pos[n]] for n in neighbours(c, shape)]) for c in combos])
        best = combos[int(np.argmax(smooth))]
        folds.append(dict(test=f"{lo}..{hi}", chosen=label(best),
                          oos_chosen=stats(mat[pos[best], test]), oos_rules=stats(mat[base, test])))
        oos_sel += list(mat[pos[best], test])
        oos_base += list(mat[base, test])
        oos_days += list(names[test])
    full = sorted((dict(label(c), **stats(mat[pos[c]])) for c in combos), key=lambda r: -r["calmar"])
    report = dict(grid={k: [str(x) for x in v] for k, v in GRID.items()}, days=len(days),
                  rules_full=stats(mat[base]), folds=folds,
                  oos_walk_forward=stats(oos_sel), oos_rules=stats(oos_base), top10_in_sample=full[:10])
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    with open(os.path.join(a.out, "oos_daily.csv"), "w") as f:
        f.write("day,walk_forward,rules\n" + "".join(
            f"{d},{x:.2f},{y:.2f}\n" for d, x, y in zip(oos_days, oos_sel, oos_base)))
    print(json.dumps({k: report[k] for k in ("rules_full", "oos_walk_forward", "oos_rules")}, indent=1))
    for fo in folds:
        print(fo["test"], fo["chosen"], "OOS chosen", round(fo["oos_chosen"]["net"]), "dd",
              round(fo["oos_chosen"]["dd"]), "| rules", round(fo["oos_rules"]["net"]), "dd",
              round(fo["oos_rules"]["dd"]))


if __name__ == "__main__":
    main()
