"""Cross-check the vectorised iron-fly engine against a plain minute-by-minute loop
on real cached days. Run: python tests/test_ironfly_reference.py"""
import os, sys, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backtests"))
import ironfly_s1 as s

def ref_fly(d, e, wings, p, legs=None):
    if legs is None:
        cw_i, pw_i = wings[0][e], wings[1][e]
        a = d["atm"][e]
        shorts = {0: a, 1: a}
    else:
        shorts = {0: legs[0][e], 1: legs[1][e]}
        cw_i, pw_i = legs[2][e], legs[3][e]
    if min(shorts[0], shorts[1], cw_i, pw_i) < 0:
        return None
    slip, px, cl, lot = p["slip_ticks"] * s.TICK, d["px"], d["close"], d["lot"]
    X = s.mi("15:35") if d["cas"] else s.mi("15:20")
    sides = {}
    for j, w in ((0, cw_i), (1, pw_i)):
        a = shorts[j]
        sell, buy = px[j, a, e] - slip, px[j, w, e] + slip
        sides[j] = dict(a=a, w=w, sell=sell, buy=buy, credit=sell - buy, stop=(1 + p["sl"]) * (sell - buy),
                        exit=None, how=None)
    if not all(x["credit"] > 0 for x in sides.values()):
        return None
    total = sides[0]["credit"] + sides[1]["credit"]
    for m in range(e, X):
        hit = []
        for j, sd in sides.items():
            if sd["exit"] is None and cl[j, sd["a"], m] - cl[j, sd["w"], m] >= sd["stop"]:
                sd["exit"], sd["how"] = m, "stop"; hit.append(j)
        if len(hit) == 1 and sides[1 - hit[0]]["exit"] is None:
            sides[1 - hit[0]]["stop"] = sides[1 - hit[0]]["credit"]
        if p["tgt"] is not None and all(sd["how"] is None for sd in sides.values()):
            v = sum(cl[j, sd["a"], m] - cl[j, sd["w"], m] for j, sd in sides.items())
            if v <= (1 - p["tgt"]) * total:
                for sd in sides.values(): sd["exit"], sd["how"] = m, "target"
        if all(sd["exit"] is not None for sd in sides.values()):
            break
    gross = chg = 0.0
    for j, sd in sides.items():
        if sd["exit"] is None:
            bb, sw = px[j, sd["a"], X] + slip, max(px[j, sd["w"], X] - slip, 0)
        else:
            bb, sw = cl[j, sd["a"], sd["exit"]] + slip, max(cl[j, sd["w"], sd["exit"]] - slip, 0)
        gross += (sd["sell"] - bb + sw - sd["buy"]) * lot
        chg += s.charges(sd["sell"] * lot, bb * lot) + s.charges(sw * lot, sd["buy"] * lot)
    stopped = all(sd["how"] == "stop" for sd in sides.values())
    second = max((sd["exit"] or X) for sd in sides.values())
    return gross - chg, stopped and gross < 0 and second < s.mi("14:50"), second

def ref_day(d, p, legs=None):
    wings = None if legs is not None else s.pick_wings(d, p["wing"])
    total = 0.0
    for e in range(s.mi("09:20"), s.mi(p["entry_last"]) + 1):
        r = ref_fly(d, e, wings, p, legs)
        if r is None: continue
        total += r[0]
        if r[1]:
            r2 = ref_fly(d, r[2] + 1, wings, p, legs)
            if r2 is not None: total += r2[0]
    return total

if __name__ == "__main__":
    days = s.load_cache()
    random.seed(1)
    worst = 0
    for d in random.sample(days, 25):
        for p in (s.DEFAULT, dict(s.DEFAULT, sl=0.35, tgt=None, wing=0.07)):
            a = sum(float(r["net"].sum()) for _, r in s.run_day(d, p))
            b = ref_day(d, p)
            worst = max(worst, abs(a - b))
            assert abs(a - b) < 1.0, (d["day"], p, a, b)
    import ironcondor_s2 as c2
    for d in random.sample(days, 15):
        legs, _ = c2.pick_condor(d)
        for p in (c2.DEFAULT, dict(c2.DEFAULT, sl=0.4, tgt=None)):
            a = sum(float(r["net"].sum()) for _, r in c2.run_day(d, p))
            b = ref_day(d, p, legs)
            worst = max(worst, abs(a - b))
            assert abs(a - b) < 1.0, ("S2", d["day"], p, a, b)
    print("ok, max |diff| =", round(worst, 4))
