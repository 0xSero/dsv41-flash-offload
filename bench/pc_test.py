#!/usr/bin/env python3
"""Prefix-cache exactness + speed: for each length L, A = P+S cold (unique cache_salt), B = P (warms salt s),
C = P+S with salt s (hit on P). Compare A vs C greedy tokens + per-token logprobs over the first 64 generated tokens,
and TTFT A vs C. Uses max_tokens only to bound the comparison window? NO: natural EOS; compare the common prefix."""
import argparse, glob, json, os, time, urllib.request, uuid
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:30141")
ap.add_argument("--lens", type=int, nargs="+", default=[4096, 32768, 131072]); ap.add_argument("--out", required=True)
a = ap.parse_args(); KEY = open(os.path.expanduser(os.environ.get("DSV41_KEY_FILE", "~/dsv41.key"))).read().strip()
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}; M = "deepseek-v4.1-flash"
def call(p, b):
    with urllib.request.urlopen(urllib.request.Request(a.url + p, data=json.dumps(b).encode(), headers=H), timeout=3600) as f: return json.loads(f.read())
text, n = [], 0
for f in sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True)):
    try: t = open(f, errors="ignore").read()
    except Exception: continue
    text.append(t); n += len(t)
    if n > 700_000: break
body = call("/tokenize", {"model": M, "prompt": "".join(text), "add_special_tokens": False})["tokens"]
head = call("/tokenize", {"model": M, "prompt": "<｜begin▁of▁sentence｜><｜User｜>", "add_special_tokens": False})["tokens"]
S = call("/tokenize", {"model": M, "prompt": "\n\nWhat programming language is the code above written in? Answer with one word.<｜Assistant｜></think>", "add_special_tokens": False})["tokens"]
def gen(ids, salt):
    b = {"model": M, "prompt": ids, "temperature": 0, "stream": True, "logprobs": 1, "cache_salt": salt}
    r = urllib.request.Request(a.url + "/v1/completions", headers=H, data=json.dumps(b).encode())
    t0 = time.perf_counter(); first = None; toks = []; lps = []
    with urllib.request.urlopen(r, timeout=7200) as f:
        for line in f:
            if not line.startswith(b"data: {"): continue
            c = json.loads(line[6:])["choices"][0]
            if c.get("text") is not None and c.get("logprobs"):
                first = first or time.perf_counter()
                toks += c["logprobs"]["tokens"]; lps += c["logprobs"]["token_logprobs"]
                if len(toks) >= 64:   # exactness probe: compare the first 64 greedy tokens, then disconnect (client side)
                    break
    return {"ttft": round(first - t0, 2), "toks": toks, "lps": lps}
res = []
for L in a.lens:
    P = head + body[:L]
    A = gen(P + S, "cold-" + uuid.uuid4().hex)
    salt = "pc-" + uuid.uuid4().hex
    gen(P + S[:1], salt)
    C = gen(P + S, salt)
    k = 0
    while k < min(len(A["toks"]), len(C["toks"])) and A["toks"][k] == C["toks"][k]: k += 1
    d = [abs(x - y) for x, y in zip(A["lps"][:k], C["lps"][:k]) if x is not None and y is not None]
    row = {"prompt": len(P + S), "ttft_cold": A["ttft"], "ttft_hit": C["ttft"], "gen_cold": len(A["toks"]), "gen_hit": len(C["toks"]),
           "identical_prefix_tokens": k, "fully_identical": A["toks"] == C["toks"], "max_abs_dlogprob": round(max(d), 5) if d else None,
           "mean_abs_dlogprob": round(sum(d) / len(d), 6) if d else None}
    res.append(row); print(json.dumps(row), flush=True); json.dump(res, open(a.out, "w"), indent=1)
print("PC DONE")
