#!/usr/bin/env python3
"""C1 context scan (TTFT-based prefill tok/s + inter-token decode tok/s, the vLLM bench serve / genai-perf method) with
NO output cap: each prompt = real text (Python stdlib source) truncated to L tokens + a short question, so the answer
ends at natural EOS. Usage: ctx_scan.py --url http://127.0.0.1:30141 --key-file ~/dsv41.key
--lens 8192 32768 65536 131072 196608 --out scan.json"""
import argparse, glob, json, os, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:30141")
ap.add_argument("--key-file", default=os.path.expanduser(os.environ.get("DSV41_KEY_FILE", "~/dsv41.key")))
ap.add_argument("--lens", type=int, nargs="+", default=[8192, 32768, 65536, 131072, 196608])
ap.add_argument("--out", required=True)
a = ap.parse_args()
KEY = open(a.key_file).read().strip()
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
MODEL = "deepseek-v4.1-flash"

def call(path, body, timeout=600):
    r = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers=H)
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())

files = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
text, n = [], 0
for f in files:
    try:
        t = open(f, errors="ignore").read()
    except Exception:
        continue
    text.append(f"\n# ===== {f} =====\n{t}"); n += len(t)
    if n > 1_200_000:
        break
corpus = "".join(text)
body_ids = call("/tokenize", {"model": MODEL, "prompt": corpus, "add_special_tokens": False}, 1200)["tokens"]
print("corpus tokens", len(body_ids), flush=True)
head = call("/tokenize", {"model": MODEL, "prompt": "<｜begin▁of▁sentence｜><｜User｜>Here is a large dump of Python source code:\n", "add_special_tokens": False})["tokens"]
q = call("/tokenize", {"model": MODEL, "prompt": "\n\nQuestion: name three modules that appear in the dump above and say in one sentence what each does.<｜Assistant｜></think>", "add_special_tokens": False})["tokens"]

res = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "method": "TTFT prefill + inter-token decode, C1, natural EOS, no max_tokens", "rows": []}
for L in a.lens:
    ids = head + body_ids[: L - len(head) - len(q)] + q
    assert len(body_ids) >= L, "corpus too small"
    req = urllib.request.Request(a.url + "/v1/completions", headers=H, data=json.dumps(
        {"model": MODEL, "prompt": ids, "temperature": 0, "stream": True, "cache_salt": os.urandom(8).hex(), "stream_options": {"include_usage": True}}).encode())
    t0 = time.perf_counter(); times = []; usage = None; finish = None; txt = ""
    with urllib.request.urlopen(req, timeout=7200) as f:
        for line in f:
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                if c.get("text"):
                    times.append(time.perf_counter()); txt += c["text"]
                finish = c.get("finish_reason") or finish
    ntok = usage["completion_tokens"] if usage else len(times)
    ttft = times[0] - t0
    dec = (ntok - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else None
    row = {"prompt_tokens": len(ids), "ttft_s": round(ttft, 2), "prefill_tok_s": round(len(ids) / ttft, 1),
           "completion_tokens": ntok, "decode_tok_s": round(dec, 2) if dec else None, "finish": finish, "answer_head": txt[:200]}
    res["rows"].append(row)
    print(json.dumps(row), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
json.dump(res, open(a.out, "w"), indent=1)
print("SCAN DONE", a.out)
