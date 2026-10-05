# CPU-tier throughput on decode-shaped jobs: physical cores only vs + SMT siblings (reuses the repro store, no GPU).
# usage (in the dsv41 image): python3 smt_bench.py "<cpus>" <threads>
import os, sys, time, json, torch, numpy as np
from torch.utils.cpp_extension import load
H, I, E = 5120, 2304, 384
IT, OT = H // 16, I // 16; B13, B2 = 2 * IT * OT * 48 * 2, OT * IT * 48 * 2
Hx = load(name="ft_tier_host", sources=["/k/ft_tier_ext.cpp"], build_directory="/build",
          extra_cflags=["-O3", "-march=native", "-std=c++17", "-I/k"], extra_ldflags=["-lpthread"], verbose=False)
cpus = []
for part in sys.argv[1].split(","):
    a, _, b = part.partition("-"); cpus += list(range(int(a), int(b or a) + 1))
nt = int(sys.argv[2]); L = 4
Hx.tier_init(nt, cpus, torch.zeros(16, dtype=torch.int64), torch.zeros(64 * H, dtype=torch.half), torch.zeros(64 * 24, dtype=torch.int32),
             torch.zeros(64 * H), -1, H, I, L, 0)
for li in range(L):
    h13 = torch.from_file(f"/bt/w13_{li}.bin", shared=True, size=(E * B13 + 4096) // 2, dtype=torch.int16)[: E * B13 // 2].view(E, 2, IT, OT, 48)
    h2 = torch.from_file(f"/bt/w2_{li}.bin", shared=True, size=(E * B2 + 4096) // 2, dtype=torch.int16)[: E * B2 // 2].view(E, OT, IT, 48)
    globals().setdefault("keep", []).append((h13, h2))
    p = torch.zeros(E, 3, dtype=torch.int64)
    for e in range(E): p[e, 0] = h13[e, 0].data_ptr(); p[e, 1] = h13[e, 1].data_ptr(); p[e, 2] = h2[e].data_ptr()
    s = lambda n: torch.full((E, n), 1.0, dtype=torch.half)
    Hx.tier_add_layer(li, p, s(H), s(I), s(H), s(I), s(I), s(H))
rng = np.random.default_rng(0); res = {}
for m, k in ((1, 3), (1, 6), (2, 6), (4, 6), (4, 10)):
    ts = []
    for it in range(300):
        li = it % L
        sel = torch.tensor([[int(e) for e in rng.choice(E, k, replace=False)] for _ in range(m)], dtype=torch.int32)
        if m > 1: sel[1:, : k // 3] = sel[0, : k // 3]       # some sharing, like real decode
        x = torch.randn(m, H) * 0.35
        t = time.perf_counter(); Hx.tier_forward(li, x, sel, torch.full((m, k), 0.15), -1); ts.append(time.perf_counter() - t)
    ts = sorted(ts[50:]); nexp = len(set(sel.view(-1).tolist()))
    res[f"m{m}k{k}"] = dict(p50_ms=round(ts[len(ts) // 2] * 1e3, 3), p90_ms=round(ts[int(len(ts) * .9)] * 1e3, 3))
print(json.dumps({"cpus": sys.argv[1], "threads": nt, **res}), flush=True)
