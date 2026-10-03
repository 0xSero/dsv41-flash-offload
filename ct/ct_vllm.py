"""DSV41 CPU expert tier for vLLM + vllm_exl3 (port of glm53/cpu_tier.py; kernels/cpu_avx2 + dsv41/ct/ft_tier_cu_v.cu).

Decode-sized MoE calls (tokens <= DSV41_CT_MAXBSZ) on layers with host-resident (EXL3_HOST_EXPERTS) experts:
  ft_split (GPU, 1 block, graph safe): cold picks chosen by a cost model go to the CPU (x rows + picks published to pinned
      host memory, seq from a device counter); the GPU selection gets them removed (id -1 -> sentinel, weight 0)
  CPU worker (C++ thread + pinned pool) computes them with ft_mul1 straight from the pinned host trellis that
      vllm_exl3 already holds for zero-copy, writes an fp32 partial
  ft_combine (GPU): waits for the worker's done flag, adds the partial into apply_exl3_fused_moe's fp32 accumulator
Larger calls (prefill) are untouched (DMA staging / zero-copy path of vllm_exl3).

Env: DSV41_CPU_TIER=1, DSV41_CT_THREADS (22), DSV41_CT_CPUS ("2-23"), DSV41_CT_MODE (-1 auto), DSV41_CT_MAXBSZ (32),
     DSV41_CT_LAYERS (40: start the worker once this many layers registered), DSV41_CT_LIMIT (10.0 swiglu clamp),
     DSV41_CT_TZC (0.58) _THIT (0.03) _A (0.11) _B (0.16) _TOK (0.2) _MAXN (64) _FORCE_N (-1), DSV41_CT_TIMEOUT_S (2),
     DSV41_CT_SELFTEST (1), DSV41_CT_STATS_EVERY (0: log stats every N combines from the Python side, eager only)
Install: import dsv41.ct.ct_vllm as ct; ct.install(vllm_exl3.exl3)
"""
import os, time, json
import torch

_KDIR = os.environ.get("DSV41_CT_KDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "kernels", "cpu_avx2"))
_HERE = os.path.dirname(os.path.abspath(__file__))
_H = None
_CU = None
S = {"layers": [], "started": False, "init": False, "selftest": None}


def _log(msg):
    print(f" -- dsv41 cpu_tier: {msg}", flush=True)


def _build():
    global _H, _CU
    if _H is not None:
        return
    from torch.utils.cpp_extension import load
    d = os.environ.get("DSV41_CT_BUILD", "/root/.cache/dsv41_ct")
    os.makedirs(d + "/h", exist_ok=True); os.makedirs(d + "/cu", exist_ok=True)
    _H = load(name="ft_tier_host", sources=[os.path.join(_KDIR, "ft_tier_ext.cpp")], build_directory=d + "/h",
              extra_cflags=["-O3", os.environ.get("DSV41_CT_MARCH", "-march=native"), "-std=c++17", "-I" + _KDIR], extra_ldflags=["-lpthread"], verbose=False)
    _CU = load(name="ft_tier_cu_v", sources=[os.path.join(_HERE, "ft_tier_cu_v.cu")], build_directory=d + "/cu",
               extra_cuda_cflags=["-O3"], verbose=False)


def _cpus():
    out = []
    for part in os.environ.get("DSV41_CT_CPUS", "2-23").split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _f(k, d):
    return float(os.environ.get(k, d))


def _init(H, I, dev):
    _build()
    maxbsz = int(os.environ.get("DSV41_CT_MAXBSZ", "32"))
    hyb = int(os.environ.get("DSV41_CT_HYB_MAX", "0"))     # hybrid DMA + CPU-tail range (maxbsz, hyb]; 0 = off
    buf = max(maxbsz, hyb)
    S.update(H=H, I=I, dev=dev, maxbsz=maxbsz, maxpicks=8 * buf, hyb_max=hyb)
    S["ctrl"] = torch.zeros(16, dtype=torch.int64).pin_memory()
    S["hx"] = torch.zeros(buf * H, dtype=torch.half).pin_memory()
    S["picks"] = torch.zeros(S["maxpicks"] * 3, dtype=torch.int32).pin_memory()
    S["hout"] = torch.zeros(buf * H, dtype=torch.float32).pin_memory()
    S["dflag"] = torch.zeros(1, dtype=torch.int64, device=dev)
    S["seqc"] = torch.zeros(1, dtype=torch.int64, device=dev)
    S["stats"] = torch.zeros(8, dtype=torch.int64, device=dev)
    S["pol"] = torch.tensor([_f("DSV41_CT_TZC", 0.58), _f("DSV41_CT_THIT", 0.03), _f("DSV41_CT_A", 0.11), _f("DSV41_CT_B", 0.16),
                             _f("DSV41_CT_TOK", 0.2), _f("DSV41_CT_MAXN", 64), _f("DSV41_CT_FORCE_N", -1), 0.0, 0.0],
                            dtype=torch.float32, device=dev)
    S["timeout_ns"] = int(_f("DSV41_CT_TIMEOUT_S", 2) * 1e9)
    S["sel"] = torch.empty(S["maxpicks"], dtype=torch.long, device=dev)
    S["w"] = torch.empty(S["maxpicks"], dtype=torch.float32, device=dev)
    # hybrid: every miss goes to the CPU (force_n = all); the "misses" are exactly the tail-batch experts (fake slotof)
    S["pol_hyb"] = torch.tensor([0.54, 0.0, 0.0, 0.0, 0.0, 4096.0, 4096.0, 0.0, 1.0], dtype=torch.float32, device=dev)
    S["stats_hyb"] = torch.zeros(8, dtype=torch.int64, device=dev)
    S["hyb_n"] = 0
    threads = int(os.environ.get("DSV41_CT_THREADS", "22"))
    nl = int(os.environ.get("DSV41_CT_LAYERS", "40"))
    _H.tier_init(threads, _cpus(), S["ctrl"], S["hx"], S["picks"], S["hout"], int(os.environ.get("DSV41_CT_MODE", "-1")),
                 H, I, nl, 0)
    lim = _f("DSV41_CT_LIMIT", 10.0)
    if lim > 0:
        _H.tier_set_limit(lim)
    S.update(init=True, threads=threads, nlayers=nl, limit=lim)
    _log(f"init H {H} I {I} maxbsz {maxbsz} threads {threads} cpus {os.environ.get('DSV41_CT_CPUS', '2-23')} clamp {lim}")


def register(layer):
    h13 = getattr(layer, "_exl3_host13", None)
    h2 = getattr(layer, "_exl3_host2", None)
    if h13 is None or h2 is None or not h13.cold or int(getattr(layer, "_exl3_cold_k", 3)) != 3:
        return
    if os.environ.get("DSV41_CPU_TIER") != "1":
        return
    dev = h13.gpu.device
    H, I = int(layer._exl3_hidden_size), int(layer._exl3_intermediate_local)
    if not S["init"]:
        _init(H, I, dev)
    li = len(S["layers"])
    if li >= S["nlayers"]:
        return
    E = int(layer._exl3_num_experts)
    ptrs = torch.zeros(E, 3, dtype=torch.int64)
    slot = torch.zeros(E, dtype=torch.int32)
    for i, e in enumerate(h13.cold):
        ptrs[e, 0] = h13.host[i, 0].data_ptr(); ptrs[e, 1] = h13.host[i, 1].data_ptr(); ptrs[e, 2] = h2.host[i].data_ptr()
        slot[e] = -1
    sc = lambda t: t.detach().reshape(E, -1).half().cpu().contiguous()
    _H.tier_add_layer(li, ptrs, sc(layer.w13_suh[:, 0]), sc(layer.w13_svh[:, 0]), sc(layer.w13_suh[:, 1]), sc(layer.w13_svh[:, 1]),
                      sc(layer.w2_suh), sc(layer.w2_svh))
    hp = torch.zeros(E, 3, dtype=torch.int64)   # device-visible (UVA) addresses of each expert's home rows
    for e in range(E):
        hp[e, 0] = h13.cuda_row(e)[0].data_ptr(); hp[e, 1] = h13.cuda_row(e)[1].data_ptr(); hp[e, 2] = h2.cuda_row(e).data_ptr()
    layer._ct = {"li": li, "E": E, "slotof": slot.to(dev), "score": torch.zeros(E, dtype=torch.float32, device=dev),
                 "counts": torch.zeros(E, dtype=torch.int32, device=dev), "home": hp}
    S["layers"].append(layer)
    if len(S["layers"]) == S["nlayers"]:
        start()


@torch.inference_mode()
def selftest():
    """CPU kernel vs the GPU fused kernel reading the same cold experts zero-copy (rel RMS of the weighted routed sum)."""
    import vllm_exl3.exl3 as X
    res = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for layer in S["layers"][:: max(1, len(S["layers"]) // 3)][:3]:
        cold = layer._exl3_host13.cold
        for mt in (1, 3):
            x = (torch.randn(mt, S["H"], generator=g) * 0.35).half()
            exps = [cold[int(i)] for i in torch.randperm(len(cold), generator=g)[:6]]
            sel = torch.tensor([exps] * mt, dtype=torch.int32)
            w = torch.full((mt, 6), 0.25, dtype=torch.float32)
            cpu = _H.tier_forward(layer._ct["li"], x.float(), sel, w, -1)
            gpu = _ORIG["fused"](x.to(S["dev"], torch.bfloat16), sel.long().to(S["dev"]), w.to(S["dev"]), layer, layer._exl3_inners, None,
                                 S["limit"] if S["limit"] > 0 else None).float().cpu()
            res.append(float((cpu - gpu).norm() / gpu.norm().clamp_min(1e-12)))
    return {"max_rel_vs_gpu": round(max(res), 6), "n": len(res)}


def start():
    if S["started"]:
        return
    if os.environ.get("DSV41_CT_SELFTEST", "1") == "1":
        try:
            S["selftest"] = selftest()
        except Exception as exc:  # never block serving on the diagnostic
            import traceback
            S["selftest"] = {"error": repr(exc), "where": traceback.format_exc().strip().splitlines()[-4:]}
        _log(f"selftest {S['selftest']}")
    cpus = _cpus()
    _H.tier_start(cpus[0])
    S["started"] = True
    _log(f"started: {len(S['layers'])} layers, worker on cpu {cpus[0]}")


_ORIG = {}
_N = {"calls": 0}


def _hyb_tail(layer, T):
    """Number of trailing DMA batches the CPU tier takes for a T-token step: balance the CPU cost of those experts
    (measured: ~DSV41_HYB_C0 + DSV41_HYB_C1*(rows-1) ms per expert, plus a per-token input cost) against the DMA stream
    of the rest (DSV41_HYB_DMA ms per expert)."""
    D = getattr(layer, "_exl3_dma", None)
    if D is None:
        return 0, None
    E = layer._ct["E"]; nb = len(D["batches"]); B = D["batches"][0][1] - D["batches"][0][0]
    rows = 6.0 * T / E
    c_cpu = _f("DSV41_HYB_C0", 0.24) + _f("DSV41_HYB_C1", 0.085) * max(0.0, rows - 1)
    c_dma = _f("DSV41_HYB_DMA", 0.54)
    fixed = _f("DSV41_HYB_TOK_US", 6.0) * T / 1000.0          # host fp16->fp32 input conversion etc. (ms)
    n_cpu = max(0.0, (E * c_dma - fixed) / (c_dma + c_cpu))
    kc = int(n_cpu // B)
    kc = max(0, min(kc, nb - 8, int(_f("DSV41_HYB_MAX_BATCHES", 24))))
    return kc, D


def _fused(x2d, ids, weights, layer, inners, expert_map, limit):
    ct = getattr(layer, "_ct", None)
    T = x2d.shape[0]
    if (ct is not None and S["started"] and expert_map is None and S.get("hyb_max", 0) and S["maxbsz"] < T <= S["hyb_max"]
            and (S.get("solo", True) or os.environ.get("DSV41_HYB_SOLO_ONLY", "1") != "1")
            and ids.numel() <= S["maxpicks"] and not torch.cuda.is_current_stream_capturing()):
        kc, D = _hyb_tail(layer, T)
        if kc > 0:
            fake = ct.setdefault("hyb_slot", {}).get(kc)
            if fake is None:
                fake = torch.zeros(ct["E"], dtype=torch.int32)
                i0 = D["batches"][len(D["batches"]) - kc][0]
                for e in D["h13"].cold[i0:]:
                    fake[int(e)] = -1
                fake = ct["hyb_slot"][kc] = fake.to(S["dev"])
            n = ids.numel()
            so, wo = S["sel"][:n], S["w"][:n]
            z = x2d if (x2d.dtype == torch.half and x2d.is_contiguous()) else x2d.half().contiguous()
            S["hx"][: z.numel()].view_as(z).copy_(z, non_blocking=True)   # copy engine, same stream, before ft_split publishes
            cnt = ct.setdefault("hyb_cnt", torch.zeros(ct["E"], dtype=torch.int32, device=S["dev"]))
            _CU.ft_split(ids.contiguous().long(), weights.contiguous().float(), z, ct["E"], fake, ct["score"], S["pol_hyb"],
                         S["ctrl"].data_ptr(), S["hx"].data_ptr(), S["picks"].data_ptr(), S["seqc"], ct["li"], so, wo,
                         S["dflag"], S["stats_hyb"], cnt)
            layer._dsv41_cpu_tail = kc
            try:
                out = _ORIG["fused"](x2d, so.view(ids.shape), wo.view(weights.shape).to(weights.dtype), layer, inners, None, limit)
            finally:
                layer._dsv41_cpu_tail = 0
            _CU.ft_combine(out, S["dflag"], S["ctrl"].data_ptr(), S["hout"].data_ptr(), S["stats_hyb"], S["timeout_ns"])
            S["hyb_n"] += 1
            if S["hyb_n"] in (1, 40, 4000):
                _log(f"hybrid: T {T} cpu tail {kc} batches (step call {S['hyb_n']})")
            return out
    if ct is None or not S["started"] or T > S["maxbsz"] or expert_map is not None or ids.numel() > S["maxpicks"]:
        return _ORIG["fused"](x2d, ids, weights, layer, inners, expert_map, limit)
    n = ids.numel()
    so, wo = S["sel"][:n], S["w"][:n]
    z = x2d if (x2d.dtype == torch.half and x2d.is_contiguous()) else x2d.half().contiguous()
    _CU.ft_split(ids.contiguous().long(), weights.contiguous().float(), z, ct["E"], ct["slotof"], ct["score"], S["pol"],
                 S["ctrl"].data_ptr(), S["hx"].data_ptr(), S["picks"].data_ptr(), S["seqc"], ct["li"], so, wo, S["dflag"], S["stats"], ct["counts"])
    out = _ORIG["fused"](x2d, so.view(ids.shape), wo.view(weights.shape).to(weights.dtype), layer, inners, expert_map, limit)
    if out.dtype != torch.float32 or not out.is_contiguous():
        raise RuntimeError("dsv41 cpu_tier: expected the fp32 contiguous routed accumulator")
    _CU.ft_combine(out, S["dflag"], S["ctrl"].data_ptr(), S["hout"].data_ptr(), S["stats"], S["timeout_ns"])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# VRAM mirror cache (DSV41_EC=1). Every expert keeps its pinned host home copy; a pool of VRAM slots mirrors the
# currently hottest (layer, expert) pairs. Between engine steps (never inside a graph replay) every DSV41_EC_EVERY
# steps: read the per-layer decode routing counts written by ft_split, decay the scores, and swap the coldest residents
# for the hottest non-residents (at most DSV41_EC_ADMIT per step). A swap is three stream-ordered operations on the
# compute stream: repoint the victim's fused-kernel pointer table entries + slotof back to its home copy, copy the new
# expert's rows into the slot, repoint the newcomer to the slot. Graphs read the pointer tables and slotof by address,
# so the next replay uses the new placement; the home copy stays valid so stale pointers can only be slower, never
# wrong. _exl3_cold_mask is left all-True on purpose: every routed expert keeps taking the group (streaming) launch,
# which reads whatever its pointer says (VRAM slot or UVA home).
# Env: DSV41_EC_EVERY (16), DSV41_EC_ADMIT (48), DSV41_EC_DECAY (0.9), DSV41_EC_MARGIN_MB (900), DSV41_EC_SLOTS (0 = all
# free VRAM minus the margin), DSV41_EC_WARMUP (64: steps before the pool is allocated, after graphs + KV exist)
EC = {"N": 0, "step": 0, "owner": [], "where": {}, "score": None, "admits": 0, "evicts": 0, "replans": 0}


def _ec_alloc():
    dev = S["dev"]
    l0 = S["layers"][0]
    h13, h2 = l0._exl3_host13, l0._exl3_host2
    n13 = h13.host[0].numel(); n2 = h2.host[0].numel()
    per = (n13 + n2) * 2
    if os.environ.get("DSV41_EC_ELASTIC", "0") == "1":
        torch.cuda.empty_cache()     # return cached prefill activations to the driver; they come back on the next prefill
    free, _ = torch.cuda.mem_get_info(dev)
    want = int(os.environ.get("DSV41_EC_SLOTS", "0"))
    n = (free - int(_f("DSV41_EC_MARGIN_MB", 900)) * 2**20) // per
    n = int(min(n, want) if want > 0 else n)
    if n <= 0:
        _log(f"EC: no VRAM for slots (free {free / 2**30:.2f} GiB)"); EC["N"] = -1; return
    EC["pool"] = torch.empty(n, n13 + n2, dtype=torch.int16, device=dev)
    EC["s13"] = EC["pool"][:, :n13].view((n,) + tuple(h13.host.shape[1:]))
    EC["s2"] = EC["pool"][:, n13:].view((n,) + tuple(h2.host.shape[1:]))
    EC["N"] = n; EC["owner"] = [None] * n; EC["free"] = list(range(n))
    L = len(S["layers"]); E = S["layers"][0]._ct["E"]
    if EC.get("score") is None:
        EC["score"] = torch.zeros(L, E, dtype=torch.float64)
    EC["cstack"] = None
    _log(f"EC: {n} VRAM slots x {per / 2**20:.1f} MiB = {n * per / 2**30:.2f} GiB (free before {free / 2**30:.2f} GiB)")


def _ec_point(layer, items):
    """items: [(e, gate_ptr, up_ptr, down_ptr, slot_or_-1)] -> stream-ordered pointer-table + slotof update."""
    if not items:
        return
    dev = S["dev"]
    idx = torch.tensor([it[0] for it in items], dtype=torch.long).pin_memory().to(dev, non_blocking=True)
    ptrs = layer._exl3_ptrs
    for key, j in (("gate_trellis", 1), ("up_trellis", 2), ("down_trellis", 3)):
        v = torch.tensor([it[j] for it in items], dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
        ptrs[key].index_copy_(0, idx, v)
    sv = torch.tensor([it[4] for it in items], dtype=torch.int32).pin_memory().to(dev, non_blocking=True)
    layer._ct["slotof"].index_copy_(0, idx, sv)


@torch.inference_mode()
def ec_step():
    if not S["started"]:
        return
    EC["step"] += 1
    every = int(os.environ.get("DSV41_CT_LOG_EVERY", "2000"))
    if every and EC["step"] % every == 0:
        _log(f"stats step {EC['step']} {json.dumps(summary())}")
    if os.environ.get("DSV41_EC", "0") != "1" or EC["N"] < 0:
        return
    if EC.get("released"):
        # rewarm only after DSV41_EC_REWARM_AFTER consecutive decode-sized steps (agent loops alternate tool results and
        # decode; rewarming between them would copy the pool back and forth)
        if EC.get("last_ntok", 0) <= int(os.environ.get("DSV41_EC_PREFILL_TOKENS", "64")):
            EC["calm"] = EC.get("calm", 0) + 1
        else:
            EC["calm"] = 0
        if EC["calm"] >= int(os.environ.get("DSV41_EC_REWARM_AFTER", "4")):
            EC["calm"] = 0
            ec_rewarm()
        return
    if EC["N"] == 0:
        if EC["step"] < int(os.environ.get("DSV41_EC_WARMUP", "64")):
            return
        _ec_alloc()
        if EC["N"] <= 0:
            return
    if EC["step"] % int(os.environ.get("DSV41_EC_EVERY", "16")):
        return
    layers = S["layers"]
    dev = S["dev"]
    main = torch.cuda.current_stream(dev)
    # 0) activate last replan's newcomers: their slot copies ran on the side stream (DSV41_EC_ASYNC=1)
    pend = EC.get("pending")
    if pend:
        main.wait_event(pend["event"])
        for li, ad in pend["items"].items():
            _ec_point(layers[li], ad)
        EC["pending"] = None
    c = torch.stack([l._ct["counts"] for l in layers]).cpu().double()   # syncs the compute stream (once per replan)
    for l in layers:
        l._ct["counts"].zero_()
    if float(c.sum()) == 0:
        return
    sc = EC["score"]; sc.mul_(_f("DSV41_EC_DECAY", 0.9)).add_(c)
    N = EC["N"]; E = sc.shape[1]
    flat = sc.view(-1)
    top = torch.topk(flat, N).indices.tolist()
    want = set(i for i in top if flat[i] > 0)
    have = EC["where"]
    hyst = _f("DSV41_EC_HYST", 1.0)
    adm = sorted((i for i in want if i not in have), key=lambda i: -float(flat[i]))[: int(os.environ.get("DSV41_EC_ADMIT", "48"))]
    if not adm:
        return
    victims = sorted((i for i in have if i not in want), key=lambda i: float(flat[i]))
    per_layer = {}
    copies = []
    for i in adm:
        if EC["free"]:
            s = EC["free"].pop()
        elif victims and float(flat[i]) > hyst * float(flat[victims[0]]):
            v = victims.pop(0); s = have.pop(v); EC["evicts"] += 1
            li, e = divmod(v, E); hp = layers[li]._ct["home"][e].tolist()
            per_layer.setdefault(li, ([], []))[0].append((e, hp[0], hp[1], hp[2], -1))
        else:
            break
        li, e = divmod(i, E)
        have[i] = s; EC["owner"][s] = i; EC["admits"] += 1
        copies.append((li, e, s))
        per_layer.setdefault(li, ([], []))[1].append((e, EC["s13"][s][0].data_ptr(), EC["s13"][s][1].data_ptr(), EC["s2"][s].data_ptr(), s))
    for li, (ev, _) in per_layer.items():          # 1) victims back to their home copies (compute stream)
        _ec_point(layers[li], ev)
    asyn = os.environ.get("DSV41_EC_ASYNC", "0") == "1"
    if asyn:
        side = EC.get("side")
        if side is None:
            side = EC["side"] = torch.cuda.Stream(dev)
        side.wait_stream(main)                      # slots are free once the repoints above (and every earlier reader) ran
        ctx = torch.cuda.stream(side)
    else:
        import contextlib
        ctx = contextlib.nullcontext()
    with ctx:                                       # 2) newcomers' rows into their slots (host -> VRAM DMA)
        for li, e, s in copies:
            h13, h2 = layers[li]._exl3_host13, layers[li]._exl3_host2
            hi = layers[li]._ct.setdefault("hidx", {i2: k for k, i2 in enumerate(h13.cold)})[e]
            EC["s13"][s].copy_(h13.host[hi], non_blocking=True)
            EC["s2"][s].copy_(h2.host[hi], non_blocking=True)
    if asyn:                                        # 3a) newcomers go live at the next replan, after their copies
        ev = torch.cuda.Event(); ev.record(side)
        EC["pending"] = {"event": ev, "items": {li: ad for li, (_, ad) in per_layer.items() if ad}}
    else:                                           # 3b) newcomers point at their slots now (stream ordered)
        for li, (_, ad) in per_layer.items():
            _ec_point(layers[li], ad)
    EC["replans"] += 1


@torch.inference_mode()
def ec_release():
    """Elastic: hand the slot pool back before a prefill-sized step (stream ordered: residents repointed home first)."""
    if EC["N"] <= 0:
        return
    layers = S["layers"]; dev = S["dev"]; main = torch.cuda.current_stream(dev)
    E = layers[0]._ct["E"]
    side = EC.get("side")
    if side is not None:
        main.wait_stream(side)
    EC["pending"] = None
    per = {}
    for i in list(EC["where"]):
        li, e = divmod(i, E); hp = layers[li]._ct["home"][e].tolist()
        per.setdefault(li, []).append((e, hp[0], hp[1], hp[2], -1))
    for li, items in per.items():
        _ec_point(layers[li], items)
    EC["keep"] = list(EC["where"])               # rewarm set for the next decode
    EC["where"] = {}; EC["owner"] = []; EC["free"] = []
    for k in ("pool", "s13", "s2"):
        EC.pop(k, None)
    EC["N"] = 0; EC["releases"] = EC.get("releases", 0) + 1
    EC["released"] = True; EC["calm"] = 0
    torch.cuda.empty_cache()   # hand the pool back to the driver too, so non-PyTorch allocations can use it


@torch.inference_mode()
def ec_rewarm():
    """After a release: re-allocate the pool (prefill activations returned to the driver first) and re-admit the kept set."""
    torch.cuda.empty_cache()
    _ec_alloc()
    EC["released"] = False
    if EC["N"] <= 0:
        return
    layers = S["layers"]; E = layers[0]._ct["E"]
    flat = EC["score"].view(-1)
    keep = sorted(EC.get("keep", []), key=lambda i: -float(flat[i]))[: EC["N"]]
    per = {}
    for i in keep:
        s_ = EC["free"].pop(); li, e = divmod(i, E)
        EC["where"][i] = s_; EC["owner"][s_] = i
        h13, h2 = layers[li]._exl3_host13, layers[li]._exl3_host2
        hi = layers[li]._ct.setdefault("hidx", {i2: k for k, i2 in enumerate(h13.cold)})[e]
        EC["s13"][s_].copy_(h13.host[hi], non_blocking=True)
        EC["s2"][s_].copy_(h2.host[hi], non_blocking=True)
        per.setdefault(li, []).append((e, EC["s13"][s_][0].data_ptr(), EC["s13"][s_][1].data_ptr(), EC["s2"][s_].data_ptr(), s_))
    for li, items in per.items():
        _ec_point(layers[li], items)
    EC["rewarms"] = EC.get("rewarms", 0) + 1


def ec_summary():
    return {"slots": EC["N"], "resident": len(EC["where"]), "admits": EC["admits"], "evicts": EC["evicts"],
            "replans": EC["replans"], "steps": EC["step"], "errors": EC.get("errors", 0),
            "releases": EC.get("releases", 0), "rewarms": EC.get("rewarms", 0)}


def summary():
    if not S["started"]:
        return None
    st = S["stats"].tolist(); h = _H.tier_stats(); calls = max(1, st[3])
    return {"calls": st[3], "hits": st[0], "misses": st[1], "cpu_experts": st[2], "cpu_per_call": round(st[2] / calls, 3),
            "gpu_wait_ms": round(st[4] / 1e6 / max(1, st[5]), 4), "timeouts": st[6], "host_jobs": h[0],
            "host_busy_ms": round(h[4] / 1e6 / max(1, h[0]), 4), "selftest": S["selftest"],
            "hit_rate": round(st[0] / max(1, st[0] + st[1]), 4), "ec": ec_summary()}


def install(X):
    if os.environ.get("DSV41_CPU_TIER") != "1" or _ORIG:
        return
    _ORIG["fused"] = X.apply_exl3_fused_moe
    X.apply_exl3_fused_moe = _fused
    orig_pwal = X.Exl3MoEMethod.process_weights_after_loading

    def pwal(self, layer, _o=orig_pwal):
        _o(self, layer)
        register(layer)
    X.Exl3MoEMethod.process_weights_after_loading = pwal
    import atexit
    atexit.register(lambda: _log(f"summary {json.dumps(summary())}"))
    import importlib
    for modname in ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner"):
        try:
            GMR = importlib.import_module(modname)
        except Exception as exc:
            _log(f"step hook: {modname} unavailable ({exc!r})")
            continue
        orig_exec = GMR.GPUModelRunner.execute_model

        def execute_model(self, *a, _o=orig_exec, **k):
            try:
                so = a[0] if a else k.get("scheduler_output")
                ntok = int(getattr(so, "total_num_scheduled_tokens", 0) or 0)
                EC["last_ntok"] = ntok
                if ntok > int(os.environ.get("DSV41_EC_PREFILL_TOKENS", "64")) and os.environ.get("DSV41_EC_ELASTIC", "0") == "1":
                    ec_release()
            except Exception as exc:
                EC["errors"] = EC.get("errors", 0) + 1
                if EC["errors"] <= 3:
                    _log(f"ec_release error {exc!r}")
            r = _o(self, *a, **k)
            try:
                ec_step()
            except Exception as exc:
                EC["errors"] = EC.get("errors", 0) + 1
                if EC["errors"] <= 3:
                    _log(f"ec_step error {exc!r}")
            return r
        GMR.GPUModelRunner.execute_model = execute_model
        _log("step hook installed on %s (EC %s)" % (modname, os.environ.get("DSV41_EC", "0")))
    install_decode_share()
    install_ctx_clamp()
    _log("installed")


# ---------------------------------------------------------------------------------------------------------------------
# Decode share during long prefills (DSV41_DECODE_SHARE=f, 0 = off). On this box every prefill step streams all routed
# experts (8-13 s per chunk) and running decodes only advance one token per such step. After a step that carried
# prefill and took T seconds, the scheduler is allowed decode-only steps for up to f*T seconds (vLLM's DP
# prefill-deferral path: running chunks and new prefills wait, decodes run), then the next prefill chunk goes.
# f=0.25 costs prefill ~20 % wall time while other streams keep decoding at ~f/(1+f) of their solo rate.
DS = {"allow": 0.0, "last_t": None, "last_prefill": False, "throttled": 0}


def install_decode_share():
    f = _f("DSV41_DECODE_SHARE", 0.0)
    if f <= 0:
        return
    import time as _t
    try:
        from vllm.v1.engine import core as C
        from vllm.v1.core.sched import scheduler as SM
    except Exception as exc:
        _log(f"decode share: unavailable ({exc!r})"); return
    cap = _f("DSV41_DECODE_SHARE_CAP_S", 10.0)
    orig_sched = SM.Scheduler.schedule

    lpt = int(_f("DSV41_LONG_PREFILL_WHEN_WAITING", 0))

    def schedule(self, throttle_prefills=False, _o=orig_sched):
        if lpt > 0:
            # cap a long prompt's chunk only while other requests wait, so they ride along in the same step; a lone
            # long prompt keeps full chunks
            prefilling = sum(1 for r in self.running if r.num_computed_tokens < r.num_prompt_tokens)
            self.scheduler_config.long_prefill_token_threshold = lpt if len(self.waiting) + prefilling >= 2 else 0
        out = _o(self, throttle_prefills)
        S["solo"] = len(self.running) + len(self.waiting) <= 1   # hybrid DMA+CPU only for a lone request
        try:
            n = out.num_scheduled_tokens
            DS["last_prefill"] = any(v > 1 for v in n.values()) if n else False
        except Exception:
            DS["last_prefill"] = False
        return out
    SM.Scheduler.schedule = schedule

    def should_throttle(self):
        now = _t.monotonic()
        if DS["last_t"] is not None:
            dt = now - DS["last_t"]
            if DS["last_prefill"]:
                DS["allow"] = min(cap, DS["allow"] + f * dt)
            else:
                DS["allow"] = max(0.0, DS["allow"] - dt)
        DS["last_t"] = now
        sch = self.scheduler
        decoding = any(not r.is_prefill_chunk and r.num_computed_tokens >= r.num_prompt_tokens for r in sch.running)
        # never hold back new arrivals: with --long-prefill-token-threshold below the chunk budget they ride along in
        # the next prefill step instead of queueing behind a long prompt
        if DS["allow"] > 0 and decoding and not sch.waiting:
            sch.prefill_capacity_bound = False
            DS["throttled"] += 1
            return True
        return False
    C.EngineCore._should_throttle_prefills = should_throttle
    _log(f"decode share installed: f {f} cap {cap} s")


# ---------------------------------------------------------------------------------------------------------------------
# Near-full-context requests (DSV41_CLAMP_MAX_TOKENS=1). vLLM validates prompt_tokens <= max_model_len - max_tokens at
# tokenization and returns HTTP 400 otherwise. Agents send max_tokens on compaction / summary calls exactly when the
# conversation is near the window (Pi: 0.8 x reserve ~= 13k with a ~250k prompt), so those calls failed. Validate the
# prompt against the whole window instead; vLLM's get_max_tokens then gives the request all the room that is left.
def install_ctx_clamp():
    if os.environ.get("DSV41_CLAMP_MAX_TOKENS", "0") != "1":
        return
    try:
        from vllm.renderers import params as P
    except Exception as exc:
        _log(f"ctx clamp: unavailable ({exc!r})"); return
    if getattr(P.TokenizeParams, "_dsv41_clamp", False):
        return

    def max_input_tokens(self):
        if self.max_total_tokens is None:
            return None
        return self.max_total_tokens - min(int(self.max_output_tokens or 0), 1)
    P.TokenizeParams.max_input_tokens = property(max_input_tokens)
    P.TokenizeParams._dsv41_clamp = True
    _log("ctx clamp installed: prompts validated against the full window; max_tokens shrinks to the room left")
