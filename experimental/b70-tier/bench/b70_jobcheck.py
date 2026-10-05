# new bt_worker.run_job (padded multi-token path) vs the old gather/index_add path, through the real mailbox
import os, sys, json, numpy as np, torch
os.environ["EXL3_MOE_LIB"] = "/w/_moe-dsv41.so"; sys.path.insert(0, "/w")
import bt_worker as B
from exl3xpu.moe_offload import ops
meta = json.load(open("/bt/meta.json"))
w = B.Worker("/bt", 600, B.checkpoint_scales("/model", [l["ckpt_layer"] for l in meta["layers"]], int(meta["E"])), ops())
X, dev, mb = w.X, w.dev, w.mb
rng = np.random.default_rng(1); worst = 0.0
for it in range(60):
    T = int(rng.integers(1, 9)); li = int(rng.integers(w.L))
    picks = []
    for t in range(T):
        if T > 1 and rng.random() < 0.15: continue                      # some tokens without B70 picks
        for e in rng.choice(w.E, int(rng.integers(1, 5)), replace=False): picks.append((t, int(e), float(rng.uniform(0.05, 0.3))))
    if not picks: picks = [(0, 5, 0.2)]
    x = (rng.standard_normal((T, B.H)) * 0.35).astype(np.float16)
    mb.x[:T] = x
    arr = np.array([(t, e, np.float32(wt).view(np.int32)) for t, e, wt in picks], np.int32); mb.picks[:len(arr)] = arr
    w.run_job(li, T, len(arr)); new = torch.from_numpy(mb.out[:T].copy())
    tok = arr[:, 0].astype(np.int64); ex = arr[:, 1]; ww_ = arr[:, 2].view(np.float32)
    td = torch.from_numpy(tok).to(dev); rows = torch.from_numpy(x).to(dev).index_select(0, td)
    y = X.moe_forward(rows, torch.from_numpy(ex.reshape(-1, 1).copy()).to(dev), torch.from_numpy(ww_.reshape(-1, 1).copy()).to(dev), w.ptrs[li], B.I, B.K, w.E)
    ref = torch.zeros(T, B.H, device=dev).index_add_(0, td, y.float()).cpu()
    rel = float((new - ref).norm() / ref.norm().clamp_min(1e-9)); worst = max(worst, rel)
    empty = [t for t in range(T) if t not in set(tok.tolist())]
    assert all(float(new[t].abs().max()) == 0.0 for t in empty), ("non-zero row for token without picks", empty)
print(json.dumps({"jobs": 60, "worst_rel": round(worst, 6)}), flush=True); print("JOBCHECK DONE", flush=True)
