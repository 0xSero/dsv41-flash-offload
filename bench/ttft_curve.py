#!/usr/bin/env python3
"""TTFT vs prompt length at C1 (small/mid prompts, the agent-turn regime). Streams 1 token's worth of timing then keeps
reading to natural EOS (no max_tokens); reports TTFT only, plus decode rate."""
import argparse, glob, json, os, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:30141")
ap.add_argument("--lens", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096]); ap.add_argument("--out", required=True)
a = ap.parse_args(); KEY = open(os.path.expanduser(os.environ.get("DSV41_KEY_FILE", "~/dsv41.key"))).read().strip()
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}; M = "deepseek-v4.1-flash"
def call(p, b):
    with urllib.request.urlopen(urllib.request.Request(a.url + p, data=json.dumps(b).encode(), headers=H), timeout=600) as f: return json.loads(f.read())
src = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob("/usr/lib/python3*/json/*.py")))
body = call("/tokenize", {"model": M, "prompt": src, "add_special_tokens": False})["tokens"]
head = call("/tokenize", {"model": M, "prompt": "<｜begin▁of▁sentence｜><｜User｜>", "add_special_tokens": False})["tokens"]
q = call("/tokenize", {"model": M, "prompt": "\nSay OK.<｜Assistant｜></think>", "add_special_tokens": False})["tokens"]
res = {"rows": []}
for L in a.lens:
    ids = head + body[: max(0, L - len(head) - len(q))] + q
    r = urllib.request.Request(a.url + "/v1/completions", headers=H, data=json.dumps({"model": M, "prompt": ids, "temperature": 0, "stream": True, "cache_salt": os.urandom(8).hex()}).encode())
    t0 = time.perf_counter(); ts = []
    with urllib.request.urlopen(r, timeout=3600) as f:
        for line in f:
            if line.startswith(b"data: {") and json.loads(line[6:])["choices"][0].get("text"): ts.append(time.perf_counter())
    row = {"prompt_tokens": len(ids), "ttft_s": round(ts[0] - t0, 3), "prefill_tok_s": round(len(ids) / (ts[0] - t0), 1), "gen": len(ts)}
    res["rows"].append(row); print(json.dumps(row), flush=True); json.dump(res, open(a.out, "w"), indent=1)
print("TTFT DONE")
