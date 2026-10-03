# Prefill profile probe (auth): one cold single-step prefill of N tokens with the torch profiler on; stops at first token.
import json, os, random, sys, time, urllib.request
U = "http://127.0.0.1:30141"; N = int(sys.argv[1]) if len(sys.argv) > 1 else 7000
K = open(os.path.expanduser(os.environ.get("DSV41_KEY_FILE", "~/dsv41.key"))).read().strip()
H = {"Content-Type": "application/json", "Authorization": "Bearer " + K}
def post(path, body=None, timeout=3000):
    with urllib.request.urlopen(urllib.request.Request(U + path, data=json.dumps(body or {}).encode(), headers=H), timeout=timeout) as f: return f.read()
def prefill(n, seed, on_first=None):
    rng = random.Random(seed); ids = [rng.randrange(1000, 128000) for _ in range(n)]
    r = urllib.request.Request(U + "/v1/completions", headers=H, data=json.dumps({"model": "deepseek-v4.1-flash", "prompt": ids, "temperature": 0, "stream": True, "cache_salt": os.urandom(8).hex()}).encode())
    t = time.time(); f = urllib.request.urlopen(r, timeout=3000)
    for line in f:
        if line.startswith(b"data:"):
            dt = time.time() - t
            if on_first: on_first()
            f.close(); return dt
print("warm 2048 ttft", round(prefill(2048, 1), 2), flush=True)
post("/start_profile")
dt = prefill(N, 2, on_first=lambda: post("/stop_profile"))
print(f"profiled {N} prefill ttft s", round(dt, 2), "tok/s", round(N / dt, 1), flush=True)
