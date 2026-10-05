#!/usr/bin/env python3
"""Long-prompt / long-task quality A/B for the DSV41 server (decode path = where the CPU tier and B70 tier run).

Tasks (temperature 0, no output caps, thinking as noted):
  needle_{8k,64k,200k}  real text (python stdlib sources) with 3 planted facts at 10/50/90 % depth; asks for all three
                        + a 5-sentence summary of the surrounding code. Score: facts recovered (0-3).
  code_lru              write a Python LRU cache class + run the asserts we append. Score: tests pass (0/1).
  code_rpn              write an RPN evaluator; scored by hidden tests. Score: pass fraction.
  math_{1..4}           word problems with known integer answers (thinking on). Score: correct (0/1).
  long_gen              open-ended 1500+ word technical essay. Score: length, repeated-4gram rate (loop/garbage
                        detector), non-ASCII garbage rate.
Every task records the full text + per-token logprobs. --conc N runs the task list N times concurrently (each copy is a
separate stream; this is how the B70 tier gets multi-token jobs). --compare REF.json reports, per task, the agreement
prefix length vs the reference (greedy streams diverge on fp noise; a quality regression shows as worse scores, not as
an earlier divergence alone), mean |dlogprob| over the shared prefix, and the score deltas.
"""
import argparse, concurrent.futures as cf, glob, json, os, re, subprocess, sys, tempfile, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:30141"); ap.add_argument("--out", required=True)
ap.add_argument("--conc", type=int, default=1); ap.add_argument("--tasks", nargs="*"); ap.add_argument("--compare")
a = ap.parse_args()
KEY = open(os.path.expanduser(os.environ.get("DSV41_KEY_FILE", "~/freetoken-exl3/dsv41/serve.key"))).read().strip()
HD = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}; M = "deepseek-v4.1-flash"


def post(path, body, timeout=7200):
    with urllib.request.urlopen(urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers=HD), timeout=timeout) as f:
        return json.loads(f.read())


SRC = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))[:4000])
FACTS = [("the vault code", "7341-KESTREL"), ("the ship's name", "Marrowind Eclipse"), ("the meeting city", "Tromsø")]


def needle_prompt(ntok):
    ids = post("/tokenize", {"model": M, "prompt": SRC[: ntok * 5], "add_special_tokens": False})["tokens"][: ntok - 400]
    text = post("/detokenize", {"model": M, "tokens": ids})["prompt"]
    for (what, val), frac in zip(FACTS, (0.1, 0.5, 0.9)):
        i = text.rfind("\n", 0, int(len(text) * frac)) + 1
        text = text[:i] + f"# NOTE: {what} is {val}.\n" + text[i:]
    return [{"role": "user", "content": text + "\n\nThree NOTE comments are hidden in the code above. Quote each one exactly. "
             "Then summarize, in five sentences, what the surrounding code does."}]


LRU_TESTS = """
c = LRUCache(2); c.put(1, 1); c.put(2, 2); assert c.get(1) == 1; c.put(3, 3); assert c.get(2) == -1
c.put(4, 4); assert c.get(1) == -1; assert c.get(3) == 3; assert c.get(4) == 4
c = LRUCache(1); c.put(5, 5); c.put(5, 6); assert c.get(5) == 6; c.put(7, 7); assert c.get(5) == -1
print("ALL_OK")
"""
RPN_TESTS = [("3 4 +", 7), ("5 1 2 + 4 * + 3 -", 14), ("2 3 4 * +", 14), ("10 2 /", 5), ("4 2 5 * + 1 3 2 * + /", 2),
             ("7 2 -", 5), ("2 3 ^", 8), ("100 10 / 5 /", 2)]
MATH = [("A train leaves at 9:40 and arrives at 13:05. A second train covers the same route 25 minutes faster. How many "
         "minutes does the second train take?", 180),
        ("How many positive integers less than 1000 are divisible by 7 but not by 11?", 130),
        ("A rectangle's perimeter is 46 and its diagonal is 17. What is its area?", 120),
        ("What is the sum of all two-digit numbers whose digits sum to 9?", 486)]


def tasks():
    t = {}
    for n, k in (("needle_8k", 8000), ("needle_64k", 64000), ("needle_200k", 200000)):
        t[n] = dict(messages=needle_prompt(k), thinking=False, kind="needle")
    t["code_lru"] = dict(messages=[{"role": "user", "content": "Write a Python class LRUCache with __init__(capacity), get(key) "
                         "(returns -1 if missing) and put(key, value), O(1) operations, no imports except collections. Reply "
                         "with one ```python code block only."}], thinking=False, kind="code_lru")
    t["code_rpn"] = dict(messages=[{"role": "user", "content": "Write a Python function rpn(expr: str) -> int that evaluates a "
                         "reverse-Polish expression with integer operands separated by spaces and operators + - * / ^ "
                         "(/ is integer division, ^ is power). Reply with one ```python code block only."}], thinking=False, kind="code_rpn")
    for i, (q, ans) in enumerate(MATH):
        t[f"math_{i + 1}"] = dict(messages=[{"role": "user", "content": q + " End with 'ANSWER: <integer>'."}], thinking=True,
                                 kind="math", answer=ans)
    t["long_gen"] = dict(messages=[{"role": "user", "content": "Write a detailed technical essay (at least 1500 words) on how "
                         "mixture-of-experts language models are served when the experts do not fit in GPU memory: caching, "
                         "offloading to CPU RAM and NVMe, scheduling, and the trade-offs between latency and throughput."}],
                         thinking=False, kind="long_gen")
    return t


def run(name, spec):
    body = {"model": M, "messages": spec["messages"], "temperature": 0, "logprobs": True,
            "chat_template_kwargs": {"thinking": spec["thinking"]}}
    t0 = time.time(); r = post("/v1/chat/completions", body); dt = time.time() - t0
    ch = r["choices"][0]; msg = ch["message"]
    lp = [(c["token"], c["logprob"]) for c in (ch.get("logprobs") or {}).get("content") or []]
    return dict(name=name, text=msg.get("content") or "", reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
                finish=ch.get("finish_reason"), usage=r["usage"], wall_s=round(dt, 1), lp=lp)


def code_of(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[0] if m else text


def pyrun(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); p = f.name
    try:
        r = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=30)
        return r.stdout + r.stderr
    except Exception as exc:
        return repr(exc)
    finally:
        os.unlink(p)


def score(spec, res):
    txt = res["text"]; k = spec["kind"]
    if k == "needle":
        return {"facts": sum(v.lower() in txt.lower() for _, v in FACTS)}
    if k == "code_lru":
        return {"pass": int("ALL_OK" in pyrun(code_of(txt) + "\n" + LRU_TESTS))}
    if k == "code_rpn":
        out = pyrun(code_of(txt) + "\n" + "\n".join(f"print(int(rpn({e!r}) == {v}))" for e, v in RPN_TESTS))
        return {"pass_frac": round(out.count("1\n") / len(RPN_TESTS), 3)}
    if k == "math":
        m = re.findall(r"ANSWER:\s*(-?\d+)", txt)
        return {"correct": int(bool(m) and int(m[-1]) == spec["answer"])}
    if k == "long_gen":
        w = txt.split(); g = [" ".join(w[i:i + 4]) for i in range(max(0, len(w) - 3))]
        return {"words": len(w), "rep4": round(1 - len(set(g)) / max(1, len(g)), 4),
                "nonascii": round(sum(ord(c) > 127 for c in txt) / max(1, len(txt)), 4)}


T = tasks()
names = a.tasks or list(T)
jobs = [(n, c) for c in range(a.conc) for n in names]
out = {"url": a.url, "conc": a.conc, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": []}
with cf.ThreadPoolExecutor(max_workers=a.conc) as ex:
    futs = {}
    for c in range(a.conc):          # each worker walks the task list in its own order, so streams overlap in kind
        order = names[c % len(names):] + names[: c % len(names)]
        futs[ex.submit(lambda o=order, c=c: [dict(run(n, T[n]), copy=c) for n in o])] = c
    for f in cf.as_completed(futs):
        for res in f.result():
            res["score"] = score(T[res["name"]], res); out["runs"].append(res)
            print(json.dumps({k: res[k] for k in ("name", "copy", "score", "finish", "wall_s")} | {"out_tok": res["usage"]["completion_tokens"]}), flush=True)
json.dump(out, open(a.out, "w"))
if a.compare:
    ref = {r["name"]: r for r in json.load(open(a.compare))["runs"] if r.get("copy", 0) == 0}
    for r in sorted(out["runs"], key=lambda r: (r["name"], r["copy"])):
        b = ref.get(r["name"])
        if not b: continue
        n = 0
        while n < min(len(r["lp"]), len(b["lp"])) and r["lp"][n][0] == b["lp"][n][0]: n += 1
        dl = sum(abs(r["lp"][i][1] - b["lp"][i][1]) for i in range(n)) / max(1, n)
        print(json.dumps({"name": r["name"], "copy": r["copy"], "agree_prefix": n, "len": len(r["lp"]), "ref_len": len(b["lp"]),
                          "mean_abs_dlogprob": round(dl, 5), "score": r["score"], "ref_score": b["score"]}), flush=True)
print("QA DONE", flush=True)
