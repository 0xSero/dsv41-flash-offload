# B70 worker multi-token job path: current (gather rows -> M=npk,k=1 launch -> index_add) vs padded (M=T, k=kmax,
# weight-0 padding with an id already in the token's list -> output [T, H] directly), and vs padded + one packed H2D.
# Runs on the repro store (/bt = /dev/shm/dsv41rep, real experts), c3:00.0 only.
import os, sys, time, json, numpy as np, torch
os.environ["EXL3_MOE_LIB"] = "/w/_moe-dsv41.so"
sys.path.insert(0, "/w")
import bt_worker as B
from exl3xpu.moe_offload import ops
meta = json.load(open("/bt/meta.json"))
w = B.Worker("/bt", 600, B.checkpoint_scales("/model", [l["ckpt_layer"] for l in meta["layers"]], int(meta["E"])), ops())
X, dev, H, I, K, E = w.X, w.dev, B.H, B.I, B.K, w.E
li = 0
w.ensure(li, np.arange(0, 128))
torch.xpu.synchronize()
rng = np.random.default_rng(0)
pin = torch.empty(3 * 64 * 8, dtype=torch.int32).pin_memory()

def job_picks(T, per_tok):
    rows = []
    for t in range(T):
        for e in rng.choice(128, per_tok, replace=False):
            rows.append((t, int(e), float(rng.uniform(0.05, 0.3))))
    return rows

def path_cur(T, picks):
    tok = np.array([p[0] for p in picks], np.int64); ex = np.array([p[1] for p in picks], np.int32); ww_ = np.array([p[2] for p in picks], np.float32)
    td = torch.from_numpy(tok).to(dev)
    rows = w.xd.index_select(0, td)
    ids = torch.from_numpy(ex.reshape(-1, 1)).to(dev)
    ww = torch.from_numpy(ww_.reshape(-1, 1).copy()).to(dev)
    y = X.moe_forward(rows, ids, ww, w.ptrs[li], I, K, E)
    w.od[:T].zero_(); w.od[:T].index_add_(0, td, y.float())
    torch.xpu.synchronize()
    return w.od[:T].clone()

def pad_arrays(T, picks):
    per = [[] for _ in range(T)]
    for t, e, wt in picks: per[t].append((e, wt))
    km = max(len(p) for p in per)
    ids = np.zeros((T, km), np.int32); ww_ = np.zeros((T, km), np.float32)
    for t, p in enumerate(per):
        for j in range(km):
            e, wt = p[j] if j < len(p) else (p[0][0], 0.0)
            ids[t, j] = e; ww_[t, j] = wt
    return ids, ww_, km

def path_pad(T, picks):
    ids_, ww_, km = pad_arrays(T, picks)
    ids = torch.from_numpy(ids_).to(dev); ww = torch.from_numpy(ww_).to(dev)
    y = X.moe_forward(w.xd[:T], ids, ww, w.ptrs[li], I, K, E)
    w.od[:T].copy_(y.float())
    torch.xpu.synchronize()
    return w.od[:T].clone()

def path_pad1(T, picks):
    ids_, ww_, km = pad_arrays(T, picks)
    n = T * km
    a = pin.numpy(); a[:n] = ids_.reshape(-1); a[n:2 * n] = ww_.reshape(-1).view(np.int32)
    d = pin[:2 * n].to(dev, non_blocking=True)
    y = X.moe_forward(w.xd[:T], d[:n].view(T, km), d[n:2 * n].view(torch.float32).view(T, km), w.ptrs[li], I, K, E)
    w.od[:T].copy_(y.float())
    torch.xpu.synchronize()
    return w.od[:T].clone()

w.xd.copy_((torch.randn(B.MAXT, H) * 0.35).half())
res = []
for T, per in ((1, 2), (1, 4), (2, 2), (2, 3), (3, 2), (4, 2), (4, 3), (4, 4), (8, 3)):
    picks = job_picks(T, per)
    a, b, c = path_cur(T, picks), path_pad(T, picks), path_pad1(T, picks)
    err = float((b - a).norm() / a.norm()); err1 = float((c - a).norm() / a.norm())
    row = dict(T=T, picks=len(picks), rel_pad=round(err, 6), rel_pad1=round(err1, 6))
    for name, fn in (("cur", path_cur), ("pad", path_pad), ("pad1", path_pad1)):
        for _ in range(10): fn(T, picks)
        ts = []
        for _ in range(200):
            t = time.perf_counter(); fn(T, picks); ts.append(time.perf_counter() - t)
        ts.sort(); row[name + "_p50_ms"] = round(ts[100] * 1e3, 3)
    res.append(row); print(json.dumps(row), flush=True)
json.dump(res, open("/out/b70_tpath.json", "w"), indent=1)
print("TPATH DONE", flush=True)
