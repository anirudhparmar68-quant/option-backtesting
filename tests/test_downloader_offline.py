import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import upstox_downloader as u
class Fake:
    def option_contracts(self, k, e):
        return [{"instrument_key":"NSE_FO|1|"+e,"strike_price":24000,"instrument_type":"CE","trading_symbol":"X","lot_size":75,"exchange_token":1}]
    def expired_candles(self, k, f, t):
        return [[t+"T09:15:00+05:30",1,2,0.5,1.5,100,10]] if t >= "2025-03-01" else []
db = u.open_db(":memory:")
u.sync_contracts(Fake(), db, "NIFTY", "x", "2025-04-24")
print(u.download_contract(Fake(), db, 1, "NSE_FO|1|2025-04-24", "2025-04-24", 120))
print(db.execute("select * from contracts").fetchall())
