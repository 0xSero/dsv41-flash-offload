#!/usr/bin/env python3
"""Decode-during-prefill probe: stream A (chat, ~800-word answer, natural EOS) starts; once A has 20 tokens, request B
(L-token real-text prompt, unique cache_salt) starts. Reports A's tok/s while B is prefilling (A tokens between B send
and B first token / that time), B's TTFT, and A's overall rate."""
import argparse, glob, json, os, threading, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:30141")
ap.add_argument("--len", type=int, default=65536); ap.add_argument("--out", required=True)
a = ap.parse_args(); KEY = open(os.path.expanduser(os.environ.get("DSV41_KEY_FILE", "~/dsv41.key"))).read().strip()
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}; M = "deepseek-v4.1-flash"
def call(p, b):
    with urllib.request.urlopen(urllib.request.Request(a.url + p, data=json.dumps(b).encode(), headers=H), timeout=3600) as f: return json.loads(f.read())
text, n = [], 0
for f in sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True)):
    try: t = open(f, errors="ignore").read()
    except Exception: continue
    text.append(t); n += len(t)
    if n > 400_000: break
body = call("/tokenize", {"model": M, "prompt": "".join(text), "add_special_tokens": False})["tokens"]
head = call("/tokenize", {"model": M, "prompt": "<｜begin▁of▁sentence｜><｜User｜>", "add_special_tokens": False})["tokens"]
q = call("/tokenize", {"model": M, "prompt": "\nWhat language is this? One word.<｜Assistant｜></think>", "add_special_tokens": False})["tokens"]
A_ts, B = [], {}
def stream_a():
    b = {"model": M, "temperature": 0, "stream": True, "cache_salt": os.urandom(8).hex(), "chat_template_kwargs": {"thinking": False},
         "messages": [{"role": "user", "content": "Write an 800-word essay on the history of railways."}]}
    with urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(b).encode(), headers=H), timeout=7200) as f:
        for line in f:
            if line.startswith(b"data: {") and (json.loads(line[6:])["choices"] or [{}])[0].get("delta", {}).get("content"): A_ts.append(time.perf_counter())
def stream_b():
    ids = head + body[: a.len] + q
    B["t0"] = time.perf_counter()
    r = urllib.request.Request(a.url + "/v1/completions", headers=H, data=json.dumps({"model": M, "prompt": ids, "temperature": 0, "stream": True, "cache_salt": os.urandom(8).hex()}).encode())
    with urllib.request.urlopen(r, timeout=7200) as f:
        for line in f:
            if line.startswith(b"data: {") and json.loads(line[6:])["choices"][0].get("text") and "t1" not in B: B["t1"] = time.perf_counter()
ta = threading.Thread(target=stream_a); ta.start()
while len(A_ts) < 20: time.sleep(0.05)
tb = threading.Thread(target=stream_b); tb.start(); tb.join(); ta.join()
during = [t for t in A_ts if B["t0"] <= t <= B["t1"]]
row = {"prefill_len": a.len, "B_ttft_s": round(B["t1"] - B["t0"], 1), "A_tok_during_B_prefill": len(during),
       "A_rate_during_B_prefill": round(len(during) / (B["t1"] - B["t0"]), 2), "A_total_tokens": len(A_ts),
       "A_rate_overall": round((len(A_ts) - 1) / (A_ts[-1] - A_ts[0]), 2)}
print(json.dumps(row)); json.dump(row, open(a.out, "w"), indent=1)
