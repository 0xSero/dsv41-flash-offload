#!/usr/bin/env python3
"""Mixed agent-like soak: W workers for M minutes; each request is a random mix of short chat, mid-size prompt
(100-3000 tokens of real text) and long prompt (16k-128k), natural EOS (no max_tokens). Counts errors and records TTFT /
decode per request. Exit 1 on any HTTP/connection error."""
import argparse, glob, json, os, random, threading, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:30141")
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--minutes", type=float, default=30); ap.add_argument("--out", required=True)
a = ap.parse_args(); KEY = open(os.path.expanduser(os.environ.get("DSV41_KEY_FILE", "~/dsv41.key"))).read().strip()
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}; M = "deepseek-v4.1-flash"
def call(p, b):
    with urllib.request.urlopen(urllib.request.Request(a.url + p, data=json.dumps(b).encode(), headers=H), timeout=1200) as f: return json.loads(f.read())
text, n = [], 0
for f in sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True)):
    try: t = open(f, errors="ignore").read()
    except Exception: continue
    text.append(t); n += len(t)
    if n > 700_000: break
body = call("/tokenize", {"model": M, "prompt": "".join(text), "add_special_tokens": False})["tokens"]
head = call("/tokenize", {"model": M, "prompt": "<｜begin▁of▁sentence｜><｜User｜>", "add_special_tokens": False})["tokens"]
q = call("/tokenize", {"model": M, "prompt": "\nSummarize the code above in two sentences.<｜Assistant｜></think>", "add_special_tokens": False})["tokens"]
rows, errs, lock = [], [], threading.Lock(); stop = time.time() + a.minutes * 60
def worker(w):
    rnd = random.Random(w)
    while time.time() < stop:
        kind = rnd.choices(["short", "mid", "long"], [3, 5, 2])[0]
        L = {"short": 0, "mid": rnd.randint(100, 3000), "long": rnd.randint(16000, 128000)}[kind]
        off = rnd.randint(0, len(body) - L - 1)
        ids = head + body[off: off + L] + q if L else head + call("/tokenize", {"model": M, "prompt": "Write a haiku about servers.", "add_special_tokens": False})["tokens"] + q[-1:]
        r = urllib.request.Request(a.url + "/v1/completions", headers=H, data=json.dumps({"model": M, "prompt": ids, "temperature": 0.7, "stream": True, "cache_salt": os.urandom(8).hex()}).encode())
        t0 = time.perf_counter(); ts = []
        try:
            with urllib.request.urlopen(r, timeout=3600) as f:
                for line in f:
                    if line.startswith(b"data: {") and json.loads(line[6:])["choices"][0].get("text"): ts.append(time.perf_counter())
            row = {"w": w, "kind": kind, "prompt": len(ids), "ttft": round(ts[0] - t0, 2) if ts else None, "gen": len(ts),
                   "dec": round((len(ts) - 1) / (ts[-1] - ts[0]), 2) if len(ts) > 2 and ts[-1] > ts[0] else None}
            with lock: rows.append(row); print(json.dumps(row), flush=True)
        except Exception as e:
            with lock: errs.append({"w": w, "kind": kind, "prompt": len(ids), "err": repr(e)[:300], "t": time.strftime("%H:%M:%S")}); print("ERR", errs[-1], flush=True)
            time.sleep(20)
th = [threading.Thread(target=worker, args=(i,)) for i in range(a.workers)]; [t.start() for t in th]; [t.join() for t in th]
json.dump({"rows": rows, "errors": errs}, open(a.out, "w"), indent=1)
print("SOAK DONE requests", len(rows), "errors", len(errs)); raise SystemExit(1 if errs else 0)
