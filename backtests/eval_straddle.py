"""Honest evaluation of the straddle search (search_straddle.py) plus an ML day filter.

1. Walk-forward OOS: each fold's choice (made on data before the fold) is run on that
   fold's quarter; the quarters are stitched. Compared with the user's rules on the
   same days.
2. ML filter (PyTorch, logistic regression on 09:15 features): trained only on days
   before each test quarter, skips days with predicted P(loss) above a threshold that is
   also fixed on training data. Applied to the user's rules.
3. Holdout (Jul-Oct 2026, never used above): the 'final' choice and the ML filter, once.
"""

import json

import numpy as np
import torch

import search_straddle as ss

RULES = dict(config=[0.2, 1.0, 1], step=1, set="BOTH", window=["09:20", "13:00"],
             filters="exp:all gap:any vix:any wd:all")


def entry_index(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m - (9 * 60 + 20)


class Data:
    def __init__(self):
        z, bars = ss.load()
        self.P = z["P"]
        self.configs = [tuple(c) for c in z["configs"]]
        self.rows = ss.features(z, bars)
        self.dates = sorted({r["day"] for r in self.rows})
        self.dpos = {d: i for i, d in enumerate(self.dates)}
        self.masks = dict(ss.filter_masks(self.rows))

    def daily(self, spec, extra_mask=None):
        """Date-level net P&L series for a spec (dict as stored by the search)."""
        c = self.configs.index(tuple(spec["config"]))
        s, e, k = entry_index(spec["window"][0]), entry_index(spec["window"][1]), spec["step"]
        ent = np.arange(s, e + 1, k)
        m = self.masks[spec["filters"]].copy()
        if extra_mask is not None:
            m &= extra_mask
        ixs = ["NIFTY", "SENSEX"] if spec["set"] == "BOTH" else [spec["set"]]
        out = np.zeros(len(self.dates))
        for i, r in enumerate(self.rows):
            if r["index"] in ixs and m[i]:
                out[self.dpos[r["day"]]] += self.P[c, i, ent].sum()
        return out


def stats(v):
    v = np.asarray(v, float)
    eq = np.cumsum(v)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq)) if len(v) else 0.0
    net = float(v.sum())
    return dict(net=round(net), maxdd=round(dd), calmar=round(net / dd, 2) if dd > 0 else None,
                win_days=round(float((v[v != 0] > 0).mean()), 3) if (v != 0).any() else None,
                active_days=int((v != 0).sum()), days=len(v))


FEATS = ["gap", "vix_open", "vix_chg", "prev_ret", "prev_range", "dte", "expiry_day"]


def ml_filter(data, train_end, test_lo, test_hi, base_spec=RULES):
    """Train on rows before train_end; return mask over rows (True = trade) for all rows,
    with the threshold chosen on the training rows (maximise train Calmar)."""
    X = np.array([[float(r[f]) if r[f] == r[f] else 0.0 for f in FEATS]
                  + [float(r["weekday"] == w) for w in range(5)]
                  + [float(r["index"] == "NIFTY")] for r in data.rows])
    # per-row P&L of the base spec (label = loss day for that index)
    c = data.configs.index(tuple(base_spec["config"]))
    s, e = entry_index(base_spec["window"][0]), entry_index(base_spec["window"][1])
    y_pnl = data.P[c, :, s:e + 1:base_spec["step"]].sum(1)
    days = np.array([r["day"] for r in data.rows])
    tr = days < train_end
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor((y_pnl < 0).astype(np.float32))
    torch.manual_seed(0)
    model = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xt[tr]).squeeze(1), yt[tr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        p_loss = torch.sigmoid(model(Xt).squeeze(1)).numpy()
    best_thr, best_cal = 1.01, -np.inf
    for thr in np.arange(0.40, 0.80, 0.02):
        keep = tr & (p_loss < thr)
        v = np.zeros(len(data.dates))
        for i in np.flatnonzero(keep):
            v[data.dpos[days[i]]] += y_pnl[i]
        st = stats(v[: np.searchsorted(data.dates, train_end)])
        if st["calmar"] is not None and st["active_days"] > 0.5 * tr.sum() / 2 and st["calmar"] > best_cal:
            best_cal, best_thr = st["calmar"], thr
    return p_loss < best_thr, best_thr


def main():
    data = Data()
    sel = json.load(open("results/straddle_search/selection.json"))
    dates = np.array(data.dates)
    report = {"combos_searched": sel["combos"], "folds": []}
    wf, base, mlv = [], [], []
    for q, (lo, hi) in enumerate(ss.FOLDS):
        spec = sel["best"][f"fold{q}"]
        te = (dates >= lo) & (dates <= hi)
        a, b = data.daily(spec)[te], data.daily(RULES)[te]
        keep, thr = ml_filter(data, lo, lo, hi)
        c = data.daily(RULES, keep)[te]
        wf += list(a); base += list(b); mlv += list(c)
        report["folds"].append(dict(test=f"{lo}..{hi}", chosen=spec, search_oos=stats(a),
                                    rules_oos=stats(b), ml_threshold=round(float(thr), 2), ml_oos=stats(c)))
    report["walk_forward_oos"] = dict(search=stats(wf), rules=stats(base), ml_filter=stats(mlv))
    ho = dates >= ss.HOLDOUT
    final = sel["best"]["final"]
    keep, thr = ml_filter(data, ss.HOLDOUT, ss.HOLDOUT, "2026-12-31")
    report["holdout"] = dict(final_choice=final, search=stats(data.daily(final)[ho]),
                             rules=stats(data.daily(RULES)[ho]), ml_threshold=round(float(thr), 2),
                             ml_filter=stats(data.daily(RULES, keep)[ho]))
    report["full_period_rules"] = stats(data.daily(RULES))
    report["full_period_final_choice_in_sample_view"] = stats(data.daily(final))
    json.dump(report, open("results/straddle_search/evaluation.json", "w"), indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
