"""Offline tests for the shared rate limiter and retry/backoff. Run: python tests/test_ratelimit.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import upstox_downloader as u


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


def patch_clock():
    c = Clock()
    u.time.time, u.time.sleep = c.time, c.sleep
    return c


def test_limiter_spaces_requests_and_is_shared():
    c = patch_clock()
    path = os.path.join(tempfile.mkdtemp(), "rl.json")
    a, b = u.RateLimiter(60, path), u.RateLimiter(60, path)  # two "processes"
    start = c.t
    for i in range(70):
        (a if i % 2 else b).wait()
    # 60/min cap shared across both: 70 calls need > 60 s
    assert c.t - start >= 60, c.t - start


def test_cooldown_blocks_everyone():
    c = patch_clock()
    path = os.path.join(tempfile.mkdtemp(), "rl.json")
    a, b = u.RateLimiter(150, path), u.RateLimiter(150, path)
    a.cooldown(100)
    start = c.t
    b.wait()
    assert c.t - start >= 100


class Resp:
    def __init__(self, code, headers=None):
        self.status_code, self.headers, self.text = code, headers or {}, ""

    def json(self):
        return {"data": ["ok"]}


class Session:
    def __init__(self, codes):
        self.codes, self.headers, self.calls = list(codes), {}, 0

    def get(self, *a, **k):
        self.calls += 1
        return Resp(self.codes.pop(0), {"Retry-After": "90"} if self.calls == 1 else None)


def test_retries_429_and_5xx_then_succeeds():
    c = patch_clock()
    path = os.path.join(tempfile.mkdtemp(), "rl.json")
    s = Session([429, 503, 200])
    api = u.Upstox("x", u.RateLimiter(150, path), s)
    start = c.t
    assert api.get("/x") == ["ok"]
    assert s.calls == 3
    assert c.t - start >= 90  # Retry-After honoured


def test_gives_up_after_max_retries():
    patch_clock()
    path = os.path.join(tempfile.mkdtemp(), "rl.json")
    api = u.Upstox("x", u.RateLimiter(150, path), Session([500] * 20))
    try:
        api.get("/x")
    except RuntimeError as e:
        assert "Giving up" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_backoff_grows_and_caps():
    waits = [u.backoff(i) for i in range(12)]
    assert waits[0] < waits[3] < waits[6]
    assert max(waits) <= 600 * 1.25


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
