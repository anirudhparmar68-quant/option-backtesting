"""Cross-check the vectorised straddle engine against straddle_920.simulate_entry on
random cached days. Run: python tests/test_straddle_fast.py"""
import os, random, sqlite3, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backtests"))
import straddle_920 as st
import straddle_fast as sf

days = sf.load()
st.SL_FILL = "sl"
db = sqlite3.connect("file:data/upstox_7delta.db?mode=ro", uri=True)
random.seed(11)
worst = 0
for d in random.sample(days, 30):
    book = st.DayBook(db, d["index"], d["expiry"], d["day"])
    for sl, mr in ((0.2, 1), (0.3, 3), (0.15, 0)):
        st.SL_PCT = sl
        ref = sum(t["net"] for m in range(st.mi("09:20"), st.mi("13:00") + 1)
                  for t in st.simulate_entry(book, d["index"], m, 0.0, None, d["X"], mr))
        fast = sf.entry_pnl(d, dict(sl=sl, cost=1.0, max_re=mr)).sum()
        worst = max(worst, abs(ref - fast))
        assert abs(ref - fast) < 1, (d["index"], d["day"], sl, mr, ref, fast)
    # conservative fills and slippage
    st.SL_PCT = 0.2
    for fill, ticks in (("close", 0), ("close", 1), ("sl", 1)):
        st.SL_FILL = fill
        ref = sum(t["net"] for m in range(st.mi("09:20"), st.mi("13:00") + 1)
                  for t in st.simulate_entry(book, d["index"], m, ticks * st.TICK, None, d["X"], 1))
        fast = sf.entry_pnl(d, dict(sl=0.2, cost=1.0, max_re=1, fill=fill, slip=ticks)).sum()
        worst = max(worst, abs(ref - fast))
        assert abs(ref - fast) < 1, (d["index"], d["day"], fill, ticks, ref, fast)
    st.SL_FILL = "sl"
print("ok, max |diff| =", round(worst, 4))
