import sys, json, numpy as np
sys.path.insert(0,"/home/user/option-backtesting/backtests")
from multiprocessing import Pool
import straddle_fast as sf, final_strategy as fs, calendar_cache as cc, call_calendar_ladder as cl
if __name__=="__main__":
    days=sf.load(); bars=json.load(open("data/daily_bars.json"))
    res=[None]*len(days)
    with Pool(4,initializer=fs._init) as p:
        for i,a,b in p.imap_unordered(fs._one,range(len(days)),chunksize=4): res[i]=a
    ccs={}
    for i,d in enumerate(days):
        if d["index"]!="NIFTY": continue
        b=bars["NIFTY"]; ds=sorted(b); k=ds.index(d["day"])
        ccs[d["day"]]=0.0 if abs(b[d["day"]]["open"]/b[ds[k-1]]["close"]-1)>0.01 else res[i]
    cal,_=cl.run(cc.load(),cl.load_spot(),1,step=100.0,exit_at="09:30",max_dte=3)
    dates=sorted(set(ccs)&set(cal)); A=np.array([ccs[d] for d in dates]); B=np.array([cal[d] for d in dates]); D=np.array(dates)
    pre=D<"2026-07-01"
    print("days",len(dates),"corr(CCS NIFTY, calendar) PRE",round(np.corrcoef(A[pre],B[pre])[0,1],3),"HOLD",round(np.corrcoef(A[~pre],B[~pre])[0,1],3))
    # corr on CCS's worst 20 days
    w=np.argsort(A[pre])[:20]; print("on CCS worst 20 PRE days: CCS sum",round(A[pre][w].sum()),"calendar sum",round(B[pre][w].sum()))
    best=None
    for lots in (0,5,10,20,30,50,75,100):
        v=A+lots*B; a,b=cl.metrics(v[pre]),cl.metrics(v[~pre]); f=cl.metrics(v)
        print(f"calendar x{lots:>3} lots | PRE net {a['net']:>11,} dd {a['max_dd']:>10,} r/dd {a['ret_dd']} sh {a['sharpe']} so {a['sortino']} | HOLD net {b['net']:>10,} sh {b['sharpe']} | FULL r/dd {f['ret_dd']} sh {f['sharpe']} so {f['sortino']}")
