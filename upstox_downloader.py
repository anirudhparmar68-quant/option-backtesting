"""Download NIFTY / SENSEX expired option data from Upstox into SQLite.

Flow: expiries -> option contracts per expiry -> 1-minute candles per contract.
Also pulls 1-minute index (spot) candles for the same date span.

Needs env var UPSTOX_ACCESS_TOKEN (Upstox Plus, Expired Instruments API).
Resumable: contracts already downloaded are skipped on re-run.

Usage:
  python upstox_downloader.py                       # NIFTY + SENSEX, all expiries
  python upstox_downloader.py --index NIFTY --expiry-from 2025-01-01
  python upstox_downloader.py --strike-min 22000 --strike-max 26000
  python upstox_downloader.py --list-expiries       # just show expiries
"""

import argparse
import os
import shutil
from datetime import datetime
import sqlite3
import sys
import time
from collections import deque
from datetime import date, timedelta
from urllib.parse import quote

import requests

BASE = "https://api.upstox.com"
INDICES = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "SENSEX": "BSE_INDEX|SENSEX",
}
CHUNK_DAYS = 28  # index 1-minute data: max ~1 month per request
OPT_CHUNK_DAYS = 365  # expired option candles: a year fits in one request


class RateLimiter:
    """Stay under Upstox limits: 50/sec, 500/min, 2000/30min (with headroom)."""

    def __init__(self, limits=((1, 10), (60, 250), (1800, 1500))):
        self.limits = limits
        self.calls = deque()

    def wait(self):
        while True:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > 1800:
                self.calls.popleft()
            sleep_for = 0.0
            for window, cap in self.limits:
                recent = [t for t in self.calls if now - t < window]
                if len(recent) >= cap:
                    sleep_for = max(sleep_for, window - (now - recent[0]) + 0.01)
            if sleep_for <= 0:
                self.calls.append(now)
                return
            time.sleep(sleep_for)


class Upstox:
    def __init__(self, token):
        self.s = requests.Session()
        self.s.headers.update({
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        })
        self.rl = RateLimiter()

    def get(self, path, params=None):
        for attempt in range(10):
            self.rl.wait()
            try:
                r = self.s.get(BASE + path, params=params, timeout=30)
            except requests.RequestException as e:
                wait = 2 ** attempt
                print(f"  network error ({e}); retry in {wait}s")
                time.sleep(wait)
                continue
            if r.status_code == 200:
                return r.json().get("data")
            if r.status_code == 429 or r.status_code >= 500:
                # 429 = a longer-window limit hit: back off up to 10 min
                wait = min(2 ** attempt * 5, 600)
                print(f"  HTTP {r.status_code}; retry in {wait}s")
                time.sleep(wait)
                continue
            if r.status_code == 401:
                sys.exit("401 Unauthorized: token missing/expired. Generate a fresh UPSTOX_ACCESS_TOKEN.")
            raise RuntimeError(f"HTTP {r.status_code} on {path}: {r.text[:300]}")
        raise RuntimeError(f"Giving up on {path} after retries")

    def expiries(self, underlying_key):
        return self.get("/v2/expired-instruments/expiries",
                        {"instrument_key": underlying_key}) or []

    def option_contracts(self, underlying_key, expiry):
        return self.get("/v2/expired-instruments/option/contract",
                        {"instrument_key": underlying_key, "expiry_date": expiry}) or []

    def expired_candles(self, instrument_key, frm, to, interval="1minute"):
        path = (f"/v2/expired-instruments/historical-candle/"
                f"{quote(instrument_key, safe='')}/{interval}/{to}/{frm}")
        data = self.get(path) or {}
        return data.get("candles") or []

    def index_candles(self, instrument_key, frm, to):
        path = (f"/v3/historical-candle/{quote(instrument_key, safe='')}"
                f"/minutes/1/{to}/{frm}")
        data = self.get(path) or {}
        return data.get("candles") or []


SCHEMA = """
CREATE TABLE IF NOT EXISTS expiries (
    underlying TEXT, expiry TEXT, contracts_done INTEGER DEFAULT 0,
    PRIMARY KEY (underlying, expiry));
CREATE TABLE IF NOT EXISTS contracts (
    id INTEGER PRIMARY KEY, instrument_key TEXT UNIQUE, underlying TEXT, expiry TEXT,
    strike REAL, option_type TEXT, trading_symbol TEXT, lot_size INTEGER,
    exchange_token TEXT, candles_done INTEGER DEFAULT 0, n_candles INTEGER);
-- ts = unix epoch seconds (IST bar start); contract_id -> contracts.id
-- open/high/low/close are INTEGER paise: divide by 100 for rupees
CREATE TABLE IF NOT EXISTS option_candles (
    contract_id INTEGER, ts INTEGER, open INTEGER, high INTEGER, low INTEGER, close INTEGER,
    volume INTEGER, oi INTEGER, PRIMARY KEY (contract_id, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS index_candles (
    underlying TEXT, ts INTEGER, open INTEGER, high INTEGER, low INTEGER, close INTEGER,
    PRIMARY KEY (underlying, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS index_days_done (
    underlying TEXT, chunk_from TEXT, chunk_to TEXT,
    PRIMARY KEY (underlying, chunk_from, chunk_to));
CREATE INDEX IF NOT EXISTS ix_contracts_exp ON contracts (underlying, expiry, strike, option_type);
"""


def open_db(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db


def candle_row(c):
    # [ts, open, high, low, close, volume, oi]
    # prices stored as integer paise (x100): about half the disk of REAL
    p = [round(x * 100) for x in c[1:5]]
    return (int(datetime.fromisoformat(c[0]).timestamp()), *p,
            c[5] if len(c) > 5 else None, c[6] if len(c) > 6 else None)


def windows_back(end, max_days, chunk=OPT_CHUNK_DAYS):
    """Yield (from, to) date windows going backward from `end`."""
    start_limit = end - timedelta(days=max_days)
    to = end
    while to > start_limit:
        frm = max(to - timedelta(days=chunk - 1), start_limit)
        yield frm, to
        to = frm - timedelta(days=1)


def download_contract(api, db, cid, key, expiry, max_days):
    total = 0
    for frm, to in windows_back(date.fromisoformat(expiry), max_days):
        candles = api.expired_candles(key, frm.isoformat(), to.isoformat())
        if not candles:
            if total:  # contract had data later on, so it was not listed earlier
                break
            continue
        db.executemany(
            "INSERT OR REPLACE INTO option_candles VALUES (?,?,?,?,?,?,?,?)",
            [(cid, *candle_row(c)) for c in candles])
        total += len(candles)
    db.execute("UPDATE contracts SET candles_done=1, n_candles=? WHERE id=?", (total, cid))
    db.commit()
    return total


def sync_contracts(api, db, name, ukey, expiry):
    done = db.execute("SELECT contracts_done FROM expiries WHERE underlying=? AND expiry=?",
                      (name, expiry)).fetchone()
    if done and done[0]:
        return
    rows = []
    for c in api.option_contracts(ukey, expiry):
        rows.append((c["instrument_key"], name, expiry, c.get("strike_price"),
                     c.get("instrument_type"), c.get("trading_symbol"),
                     c.get("lot_size"), str(c.get("exchange_token", ""))))
    db.executemany(
        "INSERT OR IGNORE INTO contracts (instrument_key, underlying, expiry, strike,"
        " option_type, trading_symbol, lot_size, exchange_token) VALUES (?,?,?,?,?,?,?,?)",
        rows)
    db.execute("INSERT OR REPLACE INTO expiries VALUES (?,?,1)", (name, expiry))
    db.commit()
    print(f"  {name} {expiry}: {len(rows)} contracts")


def download_index(api, db, name, ukey, first, last):
    frm = first
    while frm <= last:
        to = min(frm + timedelta(days=CHUNK_DAYS - 1), last)
        f, t = frm.isoformat(), to.isoformat()
        seen = db.execute("SELECT 1 FROM index_days_done WHERE underlying=? AND chunk_from=?"
                          " AND chunk_to=?", (name, f, t)).fetchone()
        if not seen:
            candles = api.index_candles(ukey, f, t)
            db.executemany("INSERT OR REPLACE INTO index_candles VALUES (?,?,?,?,?,?)",
                           [(name, *candle_row(c)[:5]) for c in candles])
            # don't mark the current (still open) window as done
            if to < date.today():
                db.execute("INSERT OR IGNORE INTO index_days_done VALUES (?,?,?)", (name, f, t))
            db.commit()
        frm = to + timedelta(days=1)


_ref_cache = {}


def near_atm(db, name, expiry, strike, pct):
    """True if strike is within pct% of the index's last close on/before expiry."""
    k = (name, expiry)
    if k not in _ref_cache:
        cutoff = int(datetime.fromisoformat(expiry + "T23:59:59+05:30").timestamp())
        row = db.execute("SELECT close / 100.0 FROM index_candles WHERE underlying=? AND ts<=?"
                         " ORDER BY ts DESC LIMIT 1", (name, cutoff)).fetchone()
        _ref_cache[k] = row[0] if row else None
    ref = _ref_cache[k]
    return ref is None or abs(strike - ref) <= ref * pct / 100


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", choices=list(INDICES), action="append",
                   help="NIFTY and/or SENSEX (default: both)")
    p.add_argument("--db", default="data/upstox_options.db")
    p.add_argument("--expiry-from", help="YYYY-MM-DD")
    p.add_argument("--expiry-to", help="YYYY-MM-DD")
    p.add_argument("--strike-min", type=float)
    p.add_argument("--strike-max", type=float)
    p.add_argument("--atm-pct", type=float,
                   help="only strikes within this %% of index close on expiry day, e.g. 8")
    p.add_argument("--max-days", type=int, default=365,
                   help="how far before expiry to fetch candles (default 365)")
    p.add_argument("--no-index", action="store_true", help="skip spot index candles")
    p.add_argument("--list-expiries", action="store_true")
    a = p.parse_args()

    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        sys.exit("Set UPSTOX_ACCESS_TOKEN env var first.")
    api = Upstox(token)

    os.makedirs(os.path.dirname(a.db) or ".", exist_ok=True)
    db = open_db(a.db)

    for name in a.index or list(INDICES):
        ukey = INDICES[name]
        exps = sorted(api.expiries(ukey))
        exps = [e for e in exps
                if (not a.expiry_from or e >= a.expiry_from)
                and (not a.expiry_to or e <= a.expiry_to)]
        print(f"{name}: {len(exps)} expiries"
              + (f" ({exps[0]} .. {exps[-1]})" if exps else ""))
        if a.list_expiries or not exps:
            if a.list_expiries:
                print("  " + ", ".join(exps))
            continue

        if not a.no_index or a.atm_pct:
            first = date.fromisoformat(exps[0]) - timedelta(days=a.max_days)
            last = min(date.fromisoformat(exps[-1]), date.today())
            download_index(api, db, name, ukey, first, last)
            print(f"{name}: index candles synced {first} .. {last}")

        for e in exps:
            sync_contracts(api, db, name, ukey, e)

        q = ("SELECT id, instrument_key, expiry, strike, option_type FROM contracts"
             " WHERE underlying=? AND candles_done=0 AND expiry BETWEEN ? AND ?")
        args = [name, exps[0], exps[-1]]
        if a.strike_min is not None:
            q += " AND strike >= ?"
            args.append(a.strike_min)
        if a.strike_max is not None:
            q += " AND strike <= ?"
            args.append(a.strike_max)
        todo = db.execute(q + " ORDER BY expiry DESC, strike, option_type", args).fetchall()
        if a.atm_pct:
            todo = [r for r in todo if near_atm(db, name, r[2], r[3], a.atm_pct)]
        print(f"{name}: {len(todo)} contracts to download")
        t0 = time.time()
        for i, (cid, key, exp, strike, ot) in enumerate(todo, 1):
            if shutil.disk_usage(os.path.dirname(os.path.abspath(a.db))).free < 2e9:
                sys.exit("Less than 2 GB disk left; stopping (re-run later to resume).")
            n = download_contract(api, db, cid, key, exp, a.max_days)
            if i % 25 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                eta = (len(todo) - i) / rate / 60 if rate else 0
                print(f"  [{i}/{len(todo)}] {exp} {strike:g}{ot}: {n} bars | ETA {eta:.0f} min")

    db.close()
    print("Done ->", a.db)


if __name__ == "__main__":
    main()
