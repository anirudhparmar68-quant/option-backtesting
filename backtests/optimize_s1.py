"""Walk-forward parameter study for Genuine S1 (backtests/ironfly_s1.py).

Anti-overfitting design:
  * Every combination in a small, pre-declared grid is run once over all days
    (daily net P&L matrix).
  * Anchored walk-forward: parameters are chosen ONLY on data before each test quarter,
    then applied unchanged to that quarter. The stitched out-of-sample (OOS) result is
    the honest number; in-sample "best" numbers are reported only for contrast.
  * Selection score = Calmar (net / max drawdown) averaged over the combination and its
    one-step neighbours in the grid, so an isolated lucky spike cannot win: a parameter
    set has to sit on a plateau.
  * The user's own rules (DEFAULT) are scored OOS on the same quarters as the benchmark.

Usage: python backtests/optimize_s1.py [--out results/s1_opt]
"""

import argparse
import itertools
import json
import os
import time

import numpy as np

import ironfly_s1 as s

GRID = {
    "sl": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    "tgt": [None, 0.10, 0.20, 0.30],
    "wing": [0.07, 0.10, 0.15],
    "entry_last": ["10:00", "11:00", "13:00"],
    "reentry": [True, False],
}
TEST_QUARTERS = [("2025-07-01", "2025-09-30"), ("2025-10-01", "2025-12-31"),
                 ("2026-01-01", "2026-03-31"), ("2026-04-01", "2026-06-30"),
                 ("2026-07-01", "2026-10-31")]


def stats(v):
    v = np.asarray(v, float)
    if not len(v):
        return dict(net=0.0, dd=0.0, calmar=0.0, win=0.0, days=0)
    eq = np.cumsum(v)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq))
    net = float(v.sum())
    return dict(net=net, dd=dd, calmar=net / dd if dd > 0 else (np.inf if net > 0 else 0.0),
                win=float((v > 0).mean()), days=len(v))


def neighbours(idx, shape):
    out = [idx]
    for ax in range(len(shape)):
        for step in (-1, 1):
            j = list(idx)
            j[ax] += step
            if 0 <= j[ax] < shape[ax]:
                out.append(tuple(j))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/s1_opt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    days = s.load_cache()
    day_names = [d["day"] for d in days]
    wings = {d["day"]: {w: s.pick_wings(d, w) for w in GRID["wing"]} for d in days}

    keys = list(GRID)
    shape = tuple(len(GRID[k]) for k in keys)
    combos = list(itertools.product(*[range(n) for n in shape]))
    mat = np.zeros((len(combos), len(days)))
    t0 = time.time()
    for ci, idx in enumerate(combos):
        p = dict(s.DEFAULT, **{k: GRID[k][i] for k, i in zip(keys, idx)})
        daily, _, _ = s.run_all(days, p, wings)
        mat[ci] = [daily[n] for n in day_names]
        if ci % 50 == 0:
            print(f"  {ci}/{len(combos)} combos  {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(a.out, "daily_matrix.npy"), mat)
    pos = {c: i for i, c in enumerate(combos)}
    days_arr = np.array(day_names)

    def label(idx):
        return {k: GRID[k][i] for k, i in zip(keys, idx)}

    base_idx = tuple(GRID[k].index(s.DEFAULT[k]) for k in keys)

    folds, oos_sel, oos_base, oos_days = [], [], [], []
    for lo, hi in TEST_QUARTERS:
        train = days_arr < lo
        test = (days_arr >= lo) & (days_arr <= hi)
        if not test.any():
            continue
        cal = np.array([stats(mat[i, train])["calmar"] for i in range(len(combos))])
        cal = np.where(np.isfinite(cal), cal, np.nanmax(cal[np.isfinite(cal)]))
        smooth = np.array([np.mean([cal[pos[n]] for n in neighbours(c, shape)]) for c in combos])
        best = combos[int(np.argmax(smooth))]
        sel = mat[pos[best], test]
        base = mat[pos[base_idx], test]
        folds.append(dict(test=f"{lo}..{hi}", chosen=label(best),
                          train_score=float(smooth[pos[best]]),
                          oos_chosen=stats(sel), oos_rules=stats(base)))
        oos_sel += list(sel)
        oos_base += list(base)
        oos_days += list(days_arr[test])

    # full-period view of every combo (in-sample, for the stability map only)
    full = [dict(label(c), **stats(mat[pos[c]])) for c in combos]
    full.sort(key=lambda r: -r["calmar"])
    report = dict(grid={k: [str(x) for x in v] for k, v in GRID.items()},
                  combos=len(combos), days=len(days),
                  rules_full=stats(mat[pos[base_idx]]),
                  folds=folds,
                  oos_walk_forward=stats(oos_sel), oos_rules=stats(oos_base),
                  top10_in_sample=full[:10])
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    with open(os.path.join(a.out, "oos_daily.csv"), "w") as f:
        f.write("day,walk_forward,rules\n")
        for d, x, y in zip(oos_days, oos_sel, oos_base):
            f.write(f"{d},{x:.2f},{y:.2f}\n")
    print(json.dumps({k: report[k] for k in ("rules_full", "oos_walk_forward", "oos_rules")},
                     indent=1, default=str))
    for fo in folds:
        print(fo["test"], fo["chosen"], "OOS chosen", round(fo["oos_chosen"]["net"]),
              "dd", round(fo["oos_chosen"]["dd"]), "| rules", round(fo["oos_rules"]["net"]),
              "dd", round(fo["oos_rules"]["dd"]))


if __name__ == "__main__":
    main()
