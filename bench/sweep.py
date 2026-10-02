#!/usr/bin/env python3
"""Campaign sweep protocol (mandatory for every speed claim, both cards). See bench/PROTOCOL.txt.

  1. warm-up (512-token request, not recorded)
  2. PREFILL 8k -> 16k -> 32k -> 64k: REPS fresh random-token prompts each (unique ids, no radix hits), TTFT-based
     tok/s; median + min/max. After each length the median is compared with bench/best_known.json[card]; if it is
     below (1 - TOL) x best known the sweep EXITS EARLY (exit code 3, status EARLY_EXIT in the JSON).
  3. DECODE concurrency C = 1,2,3,4 (+ --high list, e.g. 8 16 32): C simultaneous chat requests with distinct short
     prompts, completions run to natural EOS (no generation output cap), REPS rounds per C.
     aggregate tok/s = all completion tokens / (last token time - first first-token time); per-stream tok/s =
     (n-1)/(t_last - t_first) per request. Also C1 decode at 32k context. Early-exit check vs best known C1/C4.
  --update-best rewrites best_known.json entries this run beat (only for a full, non-early-exit run).
"""
import argparse, hashlib, json, os, random, statistics, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOPICS = ["the history of the printing press", "how a jet engine works", "the causes of the French revolution",
          "photosynthesis in C4 plants", "how TCP congestion control works", "the life cycle of stars",
          "how vaccines train the immune system", "the economics of container shipping", "how compilers optimise loops",
          "the geology of volcanoes", "the rules and strategy of chess openings", "how GPS determines position",
          "the history of the bicycle", "how noise-cancelling headphones work", "the water cycle",
          "how databases implement transactions"]


API = "sglang"
MODEL = None


class IncompleteStreamError(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt



def post(url, payload, first_only=False, timeout=7200):
    path = "/generate"
    if API == "vllm":
        path = "/v1/completions"
        payload = {"model": MODEL, "prompt": payload["input_ids"], "stream": True,
                   "temperature": payload.get("sampling_params", {}).get("temperature", 0),
                   "stream_options": {"include_usage": True}, "return_token_ids": True,
                   "add_special_tokens": False}
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); times = []; last = None; text = ""; finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            data = json.loads(body)
            if API == "vllm":
                if "error" in data:
                    raise RuntimeError(data["error"])
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("text", "")
                    finish = choices[0].get("finish_reason") or finish
                    text += delta
                    token_ids = choices[0].get("token_ids")
                    if token_ids is None:
                        raise RuntimeError("vLLM stream omitted requested token_ids")
                    times.extend([time.perf_counter()] * len(token_ids))
                last = {"text": text, "meta_info": {"finish_reason": {"type": finish}}}
                usage = data.get("usage")
                if usage is not None:
                    last["meta_info"]["completion_tokens"] = usage["completion_tokens"]
            else:
                last = data
                times.append(time.perf_counter())
            if first_only and times:
                break
    if API == "vllm" and not first_only:
        meta = (last or {}).get("meta_info", {})
        if meta.get("completion_tokens") != len(times) or finish != "stop":
            raise IncompleteStreamError(
                f"incomplete natural-EOS stream: {meta}, delivered={len(times)}",
                {"response": text, "meta_info": meta, "delivered_timestamps": len(times)})
    return t0, times, last


def tokenize(url, text):
    req = urllib.request.Request(url + "/tokenize", data=json.dumps({"prompt": text, **({"model": MODEL, "add_special_tokens": False} if API == "vllm" else {})}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["tokens"]


VOCAB_MAX = 150000


def prompt_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def rand_ids(n, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, VOCAB_MAX) for _ in range(n)]


TEMPLATE = "qwen"


def chat_ids(url, i):
    q = f"Write a detailed, well-structured explanation of {TOPICS[i % len(TOPICS)]} (variant {i})."
    if TEMPLATE == "deepseek":
        t = f"<｜begin▁of▁sentence｜><｜User｜>{q}<｜Assistant｜></think>"
    elif TEMPLATE == "glm":  # GLM-5.x, thinking off
        t = f"[gMASK]<sop><|user|>{q}<|assistant|><think></think>"
    else:
        t = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return tokenize(url, t)


def med(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs), "all": xs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30100")
    ap.add_argument("--card", required=True, help="key in best_known.json, e.g. rtx3090 / b70")
    ap.add_argument("--config", required=True, help="label: commit + env summary")
    ap.add_argument("--prefill", type=int, nargs="*", default=[8192, 16384, 32768, 65536])
    ap.add_argument("--conc", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--high", type=int, nargs="*", default=[])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dec-reps", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--no-early-exit", action="store_true")
    ap.add_argument("--prefill-only", action="store_true", help="screen prefill only; does not qualify a full P2 result")
    ap.add_argument("--prompt-seed", type=int, help="fixed random prefill prompts for paired profiles")
    ap.add_argument("--min-prefill", type=float, default=0,
                    help="abort rule: a prefill rep below this tok/s records the size and skips the larger sizes")
    ap.add_argument("--update-best", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default="qwen", choices=["qwen", "glm", "deepseek"], help="chat format of the decode prompts")
    ap.add_argument("--vocab-max", type=int, default=150000, help="upper bound for random prefill token ids")
    ap.add_argument("--api", choices=["sglang", "vllm"], default="sglang")
    a = ap.parse_args()
    if a.prefill_only and a.update_best:
        ap.error("--prefill-only cannot update best_known")
    global TEMPLATE, VOCAB_MAX, API, MODEL
    API = a.api
    TEMPLATE, VOCAB_MAX = a.template, a.vocab_max
    bk_path = os.path.join(HERE, "best_known.json")
    best_all = json.load(open(bk_path)) if os.path.exists(bk_path) else {}
    best = best_all.get(a.card, {})
    if API == "vllm":
        models = json.loads(urllib.request.urlopen(a.url + "/v1/models", timeout=30).read())["data"]
        if len(models) != 1 or not models[0].get("max_model_len"):
            raise RuntimeError("vLLM sweep requires one served model with reported max_model_len")
        MODEL = models[0]["id"]
        info = {"context_length": models[0]["max_model_len"], "served_model": models[0]}
    else:
        info = json.loads(urllib.request.urlopen(a.url + "/server_info", timeout=30).read())
    ctx_len = int(info.get("context_length") or (info.get("server_args") or {}).get("context_length") or 131072)
    res = {"card": a.card, "config": a.config, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ctx_len": ctx_len,
           "protocol": "bench/sweep.py v1", "api": API, "served_model": info.get("served_model"), "template": a.template, "tol": a.tol, "best_known_before": best,
           "server_args": {k: (info.get("server_args") or info).get(k) for k in
                           ("max_running_requests", "max_total_num_tokens", "chunked_prefill_size", "kv_cache_dtype",
                            "context_length", "mem_fraction_static", "max_mamba_cache_size")},
           "prefill": {}, "decode": {}, "status": "RUNNING"}

    def save():
        json.dump(res, open(a.out, "w"), indent=1)

    def gate(key, val):
        b = best.get(key)
        ok = b is None or val >= (1 - a.tol) * b
        res.setdefault("gates", {})[key] = {"value": round(val, 2), "best_known": b, "pass": ok}
        if not ok and not a.no_early_exit:
            res["status"] = f"EARLY_EXIT at {key}: {val:.1f} < {(1 - a.tol):.2f} x best {b}"
            save(); print(res["status"], flush=True); sys.exit(3)

    seed = a.prompt_seed if a.prompt_seed is not None else int(time.time())
    res["prompt_seed"] = seed
    res["prefill_prompt_receipts"] = []
    post(a.url, {"input_ids": rand_ids(512, seed), "sampling_params": {"temperature": 0}, "stream": True}, True)
    for n in a.prefill:
        if n + 64 > ctx_len:
            res["prefill"][str(n)] = {"skipped": f"> context {ctx_len}"}; continue
        rates = []
        for rep in range(a.reps):
            ids = rand_ids(n, seed + 1000 * n + rep)
            t0, times, _ = post(a.url, {"input_ids": ids,
                                        "sampling_params": {"temperature": 0}, "stream": True}, True)
            res["prefill_prompt_receipts"].append({"tokens": n, "rep": rep,
                "prompt_sha256": prompt_hash(ids), "ttft_s": times[0] - t0})
            rates.append(round(n / (times[0] - t0), 1))
            print(f"prefill {n}: {rates[-1]} tok/s", flush=True)
            if a.min_prefill and rates[-1] < a.min_prefill:
                break
        res["prefill"][str(n)] = med(rates); save()
        if a.min_prefill and statistics.median(rates) < a.min_prefill:
            res["prefill_aborted"] = f"prefill {n} median {statistics.median(rates):.1f} < {a.min_prefill} tok/s: larger sizes skipped"
            print(res["prefill_aborted"], flush=True); save()
            break
        gate(f"prefill_{n}", statistics.median(rates))

    if a.prefill_only:
        res["status"] = "PREFILL_ONLY_DONE"
        res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
        print("PREFILL SCREEN DONE", a.out, flush=True)
        return

    def run_conc(c, rnd, prefix=None):
        out = [None] * c
        def one(i):
            try:
                ids = (prefix or []) + chat_ids(a.url, rnd * 100 + i)
                p = {"input_ids": ids, "stream": True,
                     "sampling_params": {"temperature": 0}}
                t0, times, last = post(a.url, p)
                n = (last or {}).get("meta_info", {}).get("completion_tokens")
                if n is None:
                    raise RuntimeError("server did not report completion_tokens")
                if not times:
                    raise RuntimeError("stream returned no content")
                out[i] = {"t0": t0, "first": times[0], "last": times[-1], "tokens": n,
                          "finish": ((last or {}).get("meta_info", {}).get("finish_reason") or {}).get("type"),
                          "response": (last or {}).get("text"), "prompt_tokens": len(ids),
                          "prompt_sha256": prompt_hash(ids), "delivered_timestamps": len(times)}
            except Exception as error:
                out[i] = {"error": str(error), "error_type": type(error).__name__,
                          "stream": i, "receipt": getattr(error, "receipt", None)}
        ths = [threading.Thread(target=one, args=(i,)) for i in range(c)]
        [t.start() for t in ths]; [t.join() for t in ths]
        errors = [o for o in out if "error" in o]
        if errors:
            res["status"] = "FAILED"
            res.setdefault("failed_decode_rounds", []).append({"concurrency": c, "round": rnd, "streams": out})
            save()
            raise RuntimeError(f"decode C{c} round {rnd} failed: {[e['error'] for e in errors]}; receipts saved in {a.out}")
        tot = sum(o["tokens"] for o in out)
        span = max(o["last"] for o in out) - min(o["first"] for o in out)
        per = [(o["tokens"] - 1) / (o["last"] - o["first"]) for o in out if o["tokens"] > 1 and o["last"] > o["first"]]
        return {"aggregate": round(tot / span, 2), "per_stream_mean": round(statistics.mean(per), 2),
                "per_stream_min": round(min(per), 2), "tokens": tot,
                "finish": sorted({o["finish"] for o in out if o["finish"]}), "ttft_max": round(max(o["first"] - o["t0"] for o in out), 2), "streams": out}

    for c in list(a.conc) + list(a.high):
        rounds = [run_conc(c, r) for r in range(a.dec_reps)]
        agg = [r["aggregate"] for r in rounds]
        res["decode"][f"C{c}"] = {"aggregate": med(agg), "per_stream_mean": med([r["per_stream_mean"] for r in rounds]),
                                   "rounds": rounds}
        print(f"decode C{c}: aggregate {statistics.median(agg):.2f} tok/s, per-stream "
              f"{statistics.median([r['per_stream_mean'] for r in rounds]):.2f}", flush=True)
        save()
        if c in (1, 4):
            gate(f"decode_C{c}", statistics.median(agg))
    pre = rand_ids(32768, seed + 99) if ctx_len > 40000 else None
    if pre:
        r = [run_conc(1, 90 + i, prefix=pre) for i in range(a.dec_reps)]
        res["decode"]["C1@32k"] = {"aggregate": med([x["aggregate"] for x in r]), "rounds": r}
        print(f"decode C1@32k: {statistics.median([x['aggregate'] for x in r]):.2f} tok/s", flush=True)
    res["status"] = "DONE"; res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); save()
    if a.update_best:
        for k, g in res.get("gates", {}).items():
            if g["best_known"] is None or g["value"] > g["best_known"]:
                best_all.setdefault(a.card, {})[k] = g["value"]
        json.dump(best_all, open(bk_path, "w"), indent=1)
    print("SWEEP DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
