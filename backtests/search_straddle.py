"""Large parameter search for the staggered straddle, built to resist overfitting.

Search space (~63 million combinations):
  trade rules (128, from straddle_tensors.py): SL x survivor rule x re-entries
  index set (3): NIFTY, SENSEX, both
  entry window (153): start 09:20..12:00 and end ..13:00 on a 10-minute grid
  entry step (5): every 1, 2, 3, 5, 10 minutes
  day filters (216): expiry-day (all / skip / only) x opening gap (none / skip >0.5% /
     skip >1%) x VIX open (none / skip >18 / skip <12 / skip if up >5% vs prev close)
     x weekday (none / skip Mon..Fri)
All filters use only information available at 09:15-09:20 (index open vs previous close,
VIX open), so there is no look-ahead.

Protocol:
  * HOLDOUT = Jul-Oct 2026 is never used for any choice.
  * Walk-forward on the rest: quarterly test windows, anchored training before each.
  * Selection score on training data = min(Calmar of first half, Calmar of second half):
    a combination must be good in both halves, not just on average.
  * Activity floor: >= 10 entries on an active day and active on >= 50% of days.
  * The procedure (not one lucky combination) is what gets judged: its stitched
    walk-forward OOS result, then a single look at the holdout.
"""

import json
import sys
import time

import numpy as np
import torch

HOLDOUT = "2026-07-01"
FOLDS = [("2025-07-01", "2025-09-30"), ("2025-10-01", "2025-12-31"),
         ("2026-01-01", "2026-03-31"), ("2026-04-01", "2026-06-30")]
STEPS = [1, 2, 3, 5, 10]
STARTS = list(range(0, 161, 10))           # entry index 0 = 09:20, 220 = 13:00
WINDOWS = [(s, e) for s in STARTS for e in range(s + 10, 221, 10)] + [(s, 220) for s in STARTS]
WINDOWS = sorted(set(WINDOWS))


def hhmm(i):
    t = 9 * 60 + 20 + i
    return f"{t // 60:02d}:{t % 60:02d}"


def load():
    z = np.load("data/straddle_tensors.npz", allow_pickle=True)
    bars = json.load(open("data/daily_bars.json"))
    return z, bars


def features(z, bars):
    """Per tensor row: date, index and the 09:15 information used by the filters."""
    rows = []
    for ix, day, ed, dte in zip(z["index"], z["day"], z["expiry_day"], z["dte"]):
        b = bars[ix]
        ds = sorted(b)
        k = ds.index(day) if day in b else None
        prev = b[ds[k - 1]] if k else None
        v = bars["VIX"].get(day)
        vds = sorted(bars["VIX"])
        vk = vds.index(day) if day in bars["VIX"] else None
        vprev = bars["VIX"][vds[vk - 1]] if vk else None
        rows.append(dict(
            index=ix, day=day, expiry_day=bool(ed), dte=int(dte),
            weekday=int(np.datetime64(day).astype("datetime64[D]").view("int64") + 3) % 7,  # Mon=0
            gap=(b[day]["open"] / prev["close"] - 1) if (k and day in b) else 0.0,
            vix_open=v["open"] if v else np.nan,
            vix_chg=(v["open"] / vprev["close"] - 1) if (v and vprev) else 0.0,
            prev_ret=(prev["close"] / b[ds[k - 2]]["close"] - 1) if k and k > 1 else 0.0,
            prev_range=((prev["high"] - prev["low"]) / prev["close"]) if prev else 0.0))
    return rows


def filter_masks(rows):
    """List of (name, bool mask over rows)."""
    ed = np.array([r["expiry_day"] for r in rows])
    gap = np.abs(np.array([r["gap"] for r in rows]))
    vo = np.array([r["vix_open"] for r in rows])
    vc = np.array([r["vix_chg"] for r in rows])
    wd = np.array([r["weekday"] for r in rows])
    E = [("exp:all", np.ones_like(ed)), ("exp:skip", ~ed), ("exp:only", ed)]
    G = [("gap:any", np.ones_like(ed)), ("gap<0.5%", gap <= 0.005), ("gap<1%", gap <= 0.01)]
    V = [("vix:any", np.ones_like(ed)), ("vix<=18", ~(vo > 18)), ("vix>=12", ~(vo < 12)),
         ("vixup<5%", ~(vc > 0.05))]
    W = [("wd:all", np.ones_like(ed))] + [(f"skip:{d}", wd != i) for i, d in
                                          enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])]
    out = []
    for (a, ma) in E:
        for (b, mb) in G:
            for (c, mc) in V:
                for (d, md) in W:
                    out.append((f"{a} {b} {c} {d}", ma & mb & mc & md))
    return out


def calmar(series):
    """series (..., T) -> net, maxdd, calmar along the last axis."""
    cum = torch.cumsum(series, -1)
    peak = torch.cummax(torch.cat([torch.zeros_like(cum[..., :1]), cum], -1), -1).values[..., 1:]
    dd = (peak - cum).amax(-1)
    net = cum[..., -1]
    cal = torch.where(dd > 0, net / dd.clamp_min(1e-9), torch.where(net > 0, net * 0 + 1e6, net * 0))
    return net, dd, cal


def main():
    torch.set_num_threads(4)
    z, bars = load()
    P = z["P"]                                         # (C, R, 221)
    configs = list(z["configs"])
    rows = features(z, bars)
    dates = sorted({r["day"] for r in rows})
    dpos = {d: i for i, d in enumerate(dates)}
    T = len(dates)
    idx_rows = {ix: np.array([i for i, r in enumerate(rows) if r["index"] == ix]) for ix in ("NIFTY", "SENSEX")}
    masks = filter_masks(rows)
    F = len(masks)
    mask_mat = np.stack([m for _, m in masks]).astype(np.float32)        # (F, R)
    dates_arr = np.array(dates)

    # date slices for each fold's training halves and the final (pre-holdout) training set
    def halves(end):
        tr = np.flatnonzero(dates_arr < end)
        h = len(tr) // 2
        return tr[:h], tr[h:]

    sel_sets = {f"fold{q}": halves(lo) for q, (lo, _) in enumerate(FOLDS)}
    sel_sets["final"] = halves(HOLDOUT)

    sets = {"NIFTY": ["NIFTY"], "SENSEX": ["SENSEX"], "BOTH": ["NIFTY", "SENSEX"]}
    best = {k: (-np.inf, None) for k in sel_sets}
    W = len(WINDOWS)
    t0 = time.time()
    total = 0
    for c in range(len(configs)):
        for k in STEPS:
            # window sums with stride k: prefix per residue class
            Q = P[c].astype(np.float64).copy()                              # (R, 221)
            for e in range(k, 221):
                Q[:, e] += Q[:, e - k]
            D = np.zeros((P.shape[1], W))
            n_ent = np.zeros(W)
            for wi, (s, e) in enumerate(WINDOWS):
                last = s + ((e - s) // k) * k
                D[:, wi] = Q[:, last] - (Q[:, s - k] if s - k >= 0 else 0)
                n_ent[wi] = (e - s) // k + 1
            for sname, ixs in sets.items():
                # date-level series (F, W, T) = sum over indices of mask * D
                S = torch.zeros((F, W, T))
                act = torch.zeros((F, T))
                for ix in ixs:
                    r = idx_rows[ix]
                    di = torch.tensor([dpos[rows[i]["day"]] for i in r])
                    Dm = torch.tensor(D[r], dtype=torch.float32)                # (n, W)
                    Mm = torch.tensor(mask_mat[:, r])                           # (F, n)
                    S[:, :, di] += Mm[:, None, :] * Dm.T[None, :, :]
                    act[:, di] = torch.maximum(act[:, di], Mm)
                ok_ent = torch.tensor(n_ent >= 10)
                for name, (h1, h2) in sel_sets.items():
                    tr = np.concatenate([h1, h2])
                    active = act[:, tr].mean(-1)                                # (F,)
                    _, _, c1 = calmar(S[:, :, h1])
                    _, _, c2 = calmar(S[:, :, h2])
                    score = torch.minimum(c1, c2)
                    score = torch.where((active[:, None] >= 0.5) & ok_ent[None, :], score,
                                        torch.full_like(score, -np.inf))
                    v, flat = score.flatten().max(0)
                    if v.item() > best[name][0]:
                        fi, wi = divmod(int(flat), W)
                        best[name] = (v.item(), dict(config=list(configs[c]), step=k, set=sname,
                                                     window=[hhmm(WINDOWS[wi][0]), hhmm(WINDOWS[wi][1])],
                                                     filters=masks[fi][0]))
                total += F * W
        print(f"  config {c + 1}/{len(configs)}  combos so far {total:,}  {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(combos=total, best={k: dict(score=v[0], **v[1]) for k, v in best.items()}),
              open("results/straddle_search/selection.json", "w"), indent=1, default=str)
    print(json.dumps(best, indent=1, default=str))


if __name__ == "__main__":
    import os
    os.makedirs("results/straddle_search", exist_ok=True)
    main()
