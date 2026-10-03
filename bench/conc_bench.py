#!/usr/bin/env python3
"""Concurrency scan: C streams in parallel, each a distinct short chat prompt that ends at natural EOS (no max_tokens).
Reports per-stream TTFT/decode and aggregate output tok/s = sum(completion tokens) / wall (first send -> last token)."""
import argparse, json, os, threading, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:30141")
ap.add_argument("--key-file", default=os.path.expanduser(os.environ.get("DSV41_KEY_FILE", "~/dsv41.key")))
ap.add_argument("--conc", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--out", required=True)
a = ap.parse_args()
KEY = open(a.key_file).read().strip()
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
TOPICS = ["how TCP congestion control works", "the history of the printing press", "how a CPU branch predictor works",
          "why the sky is blue", "how vaccines train the immune system", "how B-trees index a database",
          "the causes of the French Revolution", "how GPS computes a position"]
def one(i, out):
    body = {"model": "deepseek-v4.1-flash", "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"Explain {TOPICS[i % len(TOPICS)]} in about 300 words."}],
            "chat_template_kwargs": {"thinking": False}, "cache_salt": os.urandom(8).hex()}
    t0 = time.perf_counter(); times = []; usage = None
    r = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=H)
    with urllib.request.urlopen(r, timeout=3600) as f:
        for line in f:
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                if (c.get("delta") or {}).get("content"): times.append(time.perf_counter())
    n = usage["completion_tokens"] if usage else len(times)
    out[i] = {"t0": t0, "t_first": times[0], "t_last": times[-1], "n": n, "ttft_s": round(times[0] - t0, 2),
              "decode_tok_s": round((n - 1) / (times[-1] - times[0]), 2)}
res = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": []}
for C in a.conc:
    out = {}; th = [threading.Thread(target=one, args=(i, out)) for i in range(C)]
    [t.start() for t in th]; [t.join() for t in th]
    t0 = min(v["t0"] for v in out.values()); t1 = max(v["t_last"] for v in out.values())
    tot = sum(v["n"] for v in out.values())
    row = {"C": C, "agg_out_tok_s": round(tot / (t1 - t0), 2), "total_tokens": tot, "wall_s": round(t1 - t0, 1),
           "per_stream_decode": [out[i]["decode_tok_s"] for i in range(C)], "ttft": [out[i]["ttft_s"] for i in range(C)]}
    res["rows"].append(row); print(json.dumps(row), flush=True); json.dump(res, open(a.out, "w"), indent=1)
print("CONC DONE")
