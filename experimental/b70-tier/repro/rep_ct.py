# cand11 segfault reproduction without the 3090: the CPU expert tier (k2b, with its SIGSEGV reporter) and the B70 worker
# read the same shared-tmpfs expert store, exactly as in the cand11 server, while this script plays the 3090's role:
# replays captured decode routing (C1-C4 mixes) layer by layer, posting the CPU picks to the tier's ctrl block and the
# B70 picks to the mailbox, plus occasional hybrid-prefill-sized CPU jobs (513-2048 tokens).
# Env: REP_LAYERS (4) REP_CKPT0 (3) REP_HUGE (1 = MADV_HUGEPAGE on the store, as in both crashing runs) REP_BT (1)
#      REP_MIN (minutes, 40) REP_THREADS (22) REP_CPUS (26-47)
import ctypes, json, mmap, os, sys, time, faulthandler
import numpy as np, torch
faulthandler.enable()
os.sched_setaffinity(0, {int(os.environ.get("REP_MAIN_CPU", "26"))})
H, I, E, TOPK = 5120, 2304, 384, 6
IT, OT = H // 16, I // 16
B13, B2 = 2 * IT * OT * 48 * 2, OT * IT * 48 * 2
L = int(os.environ.get("REP_LAYERS", "4")); C0 = int(os.environ.get("REP_CKPT0", "3"))
ROOT, MODEL, KD = "/bt", "/model", "/k"
HUGE = os.environ.get("REP_HUGE", "1") == "1"; BT = os.environ.get("REP_BT", "1") == "1"
MINUTES = float(os.environ.get("REP_MIN", "40"))
PAD = int(os.environ.get("REP_PAD", "0"))     # bytes of tail padding per store file (fix candidate: 4096)
def log(m): print(f"[rep {time.strftime('%H:%M:%S')}] {m}", flush=True)

from torch.utils.cpp_extension import load
Hx = load(name="ft_tier_host", sources=[f"{KD}/ft_tier_ext.cpp"], build_directory="/build",
          extra_cflags=["-O3", "-march=native", "-std=c++17", "-I" + KD], extra_ldflags=["-lpthread"], verbose=False)
log("k2b built")

# ---- store: same files / layout / mapping as exl3bt _HostExpertTrellis (from_file shared + optional MADV_HUGEPAGE)
from safetensors import safe_open
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
FH = {}
def get(n):
    f = idx[n]
    if f not in FH: FH[f] = safe_open(f"{MODEL}/{f}", "pt")
    return FH[f].get_tensor(n)
os.makedirs(ROOT, exist_ok=True)
stores, scales, layers_meta = [], [], []
t0 = time.time()
for li in range(L):
    ck = C0 + li
    paths = (f"{ROOT}/w13_{li}.bin", f"{ROOT}/w2_{li}.bin")
    if os.path.exists(f"{ROOT}/done_{li}"):          # reuse the filled store; only (un)pad the tails
        for p, nb in zip(paths, (B13, B2)): os.truncate(p, E * nb + PAD)
    else:
        for p, nb in zip(paths, (B13, B2)):
            fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o666); os.ftruncate(fd, E * nb + PAD); os.fchmod(fd, 0o666); os.close(fd)
    h13 = torch.from_file(paths[0], shared=True, size=(E * B13 + PAD) // 2, dtype=torch.int16)[: E * B13 // 2].view(E, 2, IT, OT, 48)
    h2 = torch.from_file(paths[1], shared=True, size=(E * B2 + PAD) // 2, dtype=torch.int16)[: E * B2 // 2].view(E, OT, IT, 48)
    if HUGE:
        for t in (h13, h2):
            ctypes.CDLL(None).madvise(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(t.numel() * 2), 14)
    sc = {k: torch.empty(E, n, dtype=torch.half) for k, n in (("sg", H), ("vg", I), ("su", H), ("vu", I), ("sd", I), ("vd", H))}
    fill = not os.path.exists(f"{ROOT}/done_{li}")
    for e in range(E):
        b = f"layers.{ck}.ffn.experts.{e}"
        if fill:
            h13[e, 0].copy_(get(f"{b}.w1.trellis")); h13[e, 1].copy_(get(f"{b}.w3.trellis")); h2[e].copy_(get(f"{b}.w2.trellis"))
        sc["sg"][e] = get(f"{b}.w1.suh"); sc["vg"][e] = get(f"{b}.w1.svh"); sc["su"][e] = get(f"{b}.w3.suh")
        sc["vu"][e] = get(f"{b}.w3.svh"); sc["sd"][e] = get(f"{b}.w2.suh"); sc["vd"][e] = get(f"{b}.w2.svh")
    open(f"{ROOT}/done_{li}", "w").close()
    stores.append((h13, h2)); scales.append(sc)
    layers_meta.append({"li": li, "ckpt_layer": ck, "w13": f"w13_{li}.bin", "w2": f"w2_{li}.bin"})
log(f"store {L} layers ready in {time.time() - t0:.0f} s (huge={HUGE} pad={PAD})")

# ---- CPU tier, as ct_vllm._init / register / start (DSV41_CT_MAXBSZ 512, HYB_MAX 2048 -> buf 2048)
BUF = 2048; MAXPICKS = 8 * BUF
ctrl = torch.zeros(16, dtype=torch.int64); hx = torch.zeros(BUF * H, dtype=torch.half)
picks = torch.zeros(MAXPICKS * 3, dtype=torch.int32); hout = torch.zeros(BUF * H, dtype=torch.float32)
cpus = []
for part in os.environ.get("REP_CPUS", "27-47").split(","):
    a, _, b = part.partition("-"); cpus += list(range(int(a), int(b or a) + 1))
Hx.tier_init(int(os.environ.get("REP_THREADS", "21")), cpus, ctrl, hx, picks, hout, -1, H, I, L, 0)
Hx.tier_set_limit(10.0)
for li in range(L):
    h13, h2 = stores[li]
    ptrs = torch.zeros(E, 3, dtype=torch.int64)
    for e in range(E):
        ptrs[e, 0] = h13[e, 0].data_ptr(); ptrs[e, 1] = h13[e, 1].data_ptr(); ptrs[e, 2] = h2[e].data_ptr()
    s = scales[li]
    Hx.tier_add_layer(li, ptrs, s["sg"], s["vg"], s["su"], s["vu"], s["sd"], s["vd"])
x0 = (torch.randn(4, H) * 0.35)
y = Hx.tier_forward(0, x0, torch.tensor([[1, 2, 3, 4, 5, 6]] * 4, dtype=torch.int32), torch.full((4, 6), 0.2), -1)
log(f"tier selftest out rms {float(y.pow(2).mean().sqrt()):.4f} finite {bool(torch.isfinite(y).all())}")
Hx.tier_start(cpus[0])
c_np, hx_np, pk_np = ctrl.numpy(), hx.numpy(), picks.numpy().reshape(-1, 3)

# ---- B70 mailbox (same offsets as ct_vllm.bt_init / bt_worker.Mailbox)
MAXT, MAXP = 64, 512
off_x = 4096; off_p = off_x + MAXT * H * 2; off_o = off_p + MAXP * 3 * 4; off_r = off_o + MAXT * H * 4
size = (off_r + L * E + 4095) // 4096 * 4096
if BT:
    fd = os.open(f"{ROOT}/mbox.bin", os.O_RDWR | os.O_CREAT, 0o666); os.ftruncate(fd, size); os.fchmod(fd, 0o666)
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE); os.close(fd)
    mb = np.frombuffer(mm, dtype=np.uint8); mb[:] = 0
    bc = mb[:128].view(np.int64); bx = mb[off_x:off_p].view(np.float16).reshape(MAXT, H)
    bp = mb[off_p:off_o].view(np.int32).reshape(MAXP, 3)
    json.dump({"E": E, "H": H, "I": I, "MAXT": MAXT, "MAXP": MAXP, "layers": layers_meta}, open(f"{ROOT}/meta.json.tmp", "w"))
    os.replace(f"{ROOT}/meta.json.tmp", f"{ROOT}/meta.json")
    log("mailbox + meta written; waiting for the B70 worker")
    t1 = time.time()
    while bc[6] != 1:
        time.sleep(0.2)
        if time.time() - t1 > 600: log("B70 worker never became ready"); sys.exit(3)
    log(f"B70 worker ready after {time.time() - t1:.0f} s")

# ---- replay
R = np.load("/routes.npy")             # [N, 52]: li, bsz, nn, cnt, sel[48]
R = R[(R[:, 1] >= 1) & (R[:, 1] <= 4)]
log(f"{len(R)} route records")
rng = np.random.default_rng(int(time.time()))
xr = (rng.standard_normal((BUF, H)) * 0.35).astype(np.float16)
seq_c = [0]; seq_b = [0]
st = dict(steps=0, cpu_jobs=0, cpu_picks=0, bt_jobs=0, bt_picks=0, big=0, cpu_wait=0.0, bt_wait=0.0)

def post_cpu(li, ntok, pk):
    n = len(pk)
    if ntok <= 64: hx_np[: ntok * H] = xr[:ntok].reshape(-1)
    else: hx_np[: ntok * H] = xr[:ntok].reshape(-1)
    if n: pk_np[:n] = pk
    c_np[2] = li; c_np[3] = ntok; c_np[4] = n
    seq_c[0] += 1; c_np[0] = seq_c[0]

def wait_cpu():
    t = time.perf_counter()
    while c_np[1] != seq_c[0]:
        if time.perf_counter() - t > 30: log("CPU tier job timeout (30 s)"); sys.exit(4)
    st["cpu_wait"] += time.perf_counter() - t

def post_bt(li, ntok, pk):
    bx[:ntok] = xr[:ntok]; bp[:len(pk)] = pk
    bc[2] = li; bc[3] = ntok; bc[4] = len(pk)
    seq_b[0] += 1; bc[0] = seq_b[0]

def wait_bt():
    t = time.perf_counter()
    while bc[1] != seq_b[0]:
        if time.perf_counter() - t > 30: log("B70 job timeout (30 s)"); sys.exit(5)
    st["bt_wait"] += time.perf_counter() - t

deadline = time.time() + MINUTES * 60
i = int(rng.integers(len(R))); last_log = time.time(); wbits = np.float32(0.15).view(np.int32)
while time.time() < deadline:
    rec = R[i % len(R)]; i += 1
    li = int(rec[0]) % L; bsz = int(rec[1]); nn = int(rec[2]); sel = rec[4:4 + nn]
    u = {}
    for s in range(nn):
        e = int(sel[s])
        if 0 <= e < E: u.setdefault(e, []).append(s // TOPK)
    cls = {e: (2 if r < 0.35 else 1 if r < 0.62 else 0) for e, r in zip(u, rng.random(len(u)))}
    cp = np.array([(t, e, wbits) for e, ts in u.items() if cls[e] == 1 for t in ts], dtype=np.int32).reshape(-1, 3)
    bp_ = np.array([(t, e, wbits) for e, ts in u.items() if cls[e] == 2 for t in ts], dtype=np.int32).reshape(-1, 3)
    if len(cp): post_cpu(li, bsz, cp)
    if BT and len(bp_): post_bt(li, bsz, bp_)
    if len(cp): wait_cpu(); st["cpu_jobs"] += 1; st["cpu_picks"] += len(cp)
    if BT and len(bp_): wait_bt(); st["bt_jobs"] += 1; st["bt_picks"] += len(bp_)
    st["steps"] += 1
    if st["steps"] % 6000 == 0:      # hybrid prefill chunk: 513-2048 tokens, CPU takes the tail experts
        T = int(rng.integers(513, 2049))
        sel2 = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(T)])
        tail = set(range(E - int(rng.integers(40, 160)), E))
        pk = np.array([(t, int(e), wbits) for t in range(T) for e in sel2[t] if int(e) in tail], dtype=np.int32).reshape(-1, 3)
        post_cpu(int(rng.integers(L)), T, pk); wait_cpu(); st["big"] += 1
    if time.time() - last_log > 60:
        last_log = time.time()
        log(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.items()}) + f" tier {Hx.tier_stats()[:4]}")
log("DONE no crash " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.items()}))
if BT: bc[6] = 0
Hx.tier_stop()
