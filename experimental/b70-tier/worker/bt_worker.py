#!/usr/bin/env python3
"""DSV41 B70 expert tier (BT) worker.

The 3090 server keeps every routed expert's trellis in host memory that lives in shared tmpfs files (one pair of files per
MoE layer, rows in the server's layout: w13 [2, H/16, I/16, 48] int16, w2 [I/16, H/16, 48] int16). This worker maps the
same files, imports them into Level Zero (DMA at link speed, no second copy), and serves expert jobs for the B70:

  mailbox (shared file, written by the server's GPU through UVA):
    ctrl int64[16]: [0] job seq (written last), [1] done seq (worker), [2] layer li, [3] T tokens, [4] npicks,
                    [5] worker heartbeat, [6] worker state (1 = ready), [8] resident-table generation
    x    fp16 [MAXT, H]          token inputs of the MoE call
    picks int32 [MAXP, 3]        (token, expert, weight fp32 bits)
    out  fp32 [MAXT, H]          sum over picks of w * expert(x_token)
    res  int8 [L, E]             residency hint table (1 = expert resident in a B70 slot), written by the worker
  For each job: picks whose expert is not resident are staged (DMA host row -> staging, PLANAR4 conversion into an LRU
  slot) and become resident (write-through admission); then one EXL3 MoE launch computes every pick from its slot.

Run inside the XPU image (exl3xpu + the DSV41 MoE kernel .so), one B70 only.
"""
import ctypes, json, mmap, os, sys, time
from collections import OrderedDict
import numpy as np
import torch

H, I, KW, K = 5120, 2304, 48, 3
IT, OT = H // 16, I // 16                     # 320, 144 tiles
B13 = 2 * IT * OT * KW * 2                    # bytes of one w13 row (gate+up)   8,847,360
B2 = OT * IT * KW * 2                         # bytes of one w2 row               4,423,680
NSC = 2 * H + 2 * I + I + H                   # fp16 scales per expert (suh/svh)  22,272
BLOB = B13 + B2 + NSC * 2                     # 13,315,584
MAXT, MAXP = 64, 512


def s64(p):
    p = int(p)
    return p - (1 << 64) if p >= (1 << 63) else p


class Mailbox:
    CTRL = 16 * 8

    def __init__(self, path, L, E, create=False):
        self.L, self.E = L, E
        self.off_x = 4096
        self.off_p = self.off_x + MAXT * H * 2
        self.off_o = self.off_p + MAXP * 3 * 4
        self.off_r = self.off_o + MAXT * H * 4
        self.size = (self.off_r + L * E + 4095) // 4096 * 4096
        fd = os.open(path, os.O_RDWR | (os.O_CREAT if create else 0), 0o666)
        if create:
            os.ftruncate(fd, self.size)
        self.mm = mmap.mmap(fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        self.buf = np.frombuffer(self.mm, dtype=np.uint8)
        self.ctrl = self.buf[:self.CTRL].view(np.int64)
        self.x = self.buf[self.off_x:self.off_p].view(np.float16).reshape(MAXT, H)
        self.picks = self.buf[self.off_p:self.off_o].view(np.int32).reshape(MAXP, 3)
        self.out = self.buf[self.off_o:self.off_r].view(np.float32).reshape(MAXT, H)
        self.res = self.buf[self.off_r:self.off_r + L * E].view(np.int8).reshape(L, E)
        self.base = self.buf.ctypes.data


class ZeImport:
    """zexDriverImportExternalPointer: lets SYCL memcpy DMA from host memory this process did not allocate."""

    def __init__(self):
        ze = ctypes.CDLL("libze_loader.so.1")
        ze.zeInit(0)
        n = ctypes.c_uint32(0); ze.zeDriverGet(ctypes.byref(n), None)
        d = (ctypes.c_void_p * n.value)(); ze.zeDriverGet(ctypes.byref(n), d)
        self.drv = ctypes.c_void_p(d[0])
        f = ctypes.c_void_p()
        rc = ze.zeDriverGetExtensionFunctionAddress(self.drv, b"zexDriverImportExternalPointer", ctypes.byref(f))
        if rc or not f.value:
            raise RuntimeError(f"zexDriverImportExternalPointer unavailable rc={rc}")
        self.fn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)(f.value)

    def __call__(self, ptr, size):
        rc = self.fn(self.drv, ctypes.c_void_p(ptr), ctypes.c_size_t(size))
        if rc:
            raise RuntimeError(f"import failed rc={rc:#x} ptr={ptr:#x} size={size}")


def map_file(path, ro=True):
    fd = os.open(path, os.O_RDONLY if ro else os.O_RDWR)
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ if ro else mmap.PROT_READ | mmap.PROT_WRITE)
    os.close(fd)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(mm)) if not ro else None
    if addr is None:   # read-only mmap: get the address through numpy
        addr = np.frombuffer(mm, dtype=np.uint8).ctypes.data
    return mm, addr, size


def planar4(tr):
    r, n, _ = tr.shape
    w = tr.contiguous().view(torch.int32).view(r, n // 4, 4, 8, K)
    return w.permute(0, 1, 4, 2, 3).contiguous()


class Worker:
    def __init__(self, root, n_slots, scales_fn, X, log=lambda m: print(m, flush=True)):
        """root: dir with meta.json, mbox.bin, w13_<li>.bin, w2_<li>.bin. scales_fn(li) -> fp16 [E, NSC] CPU tensor in
        blob order (gate.suh, up.suh, gate.svh, up.svh, down.suh, down.svh)."""
        self.log = log
        self.meta = json.load(open(os.path.join(root, "meta.json")))
        self.L, self.E = len(self.meta["layers"]), int(self.meta["E"])
        self.X = X
        self.dev = torch.device("xpu")
        self.imp = ZeImport()
        self.mb = Mailbox(os.path.join(root, "mbox.bin"), self.L, self.E)
        self.imp(self.mb.base, self.mb.size)
        self.h13, self.h2, self.keep = [], [], []
        t0 = time.time()
        for li in range(self.L):
            for name, lst, rb in (("w13", self.h13, B13), ("w2", self.h2, B2)):
                fn = self.meta["layers"][li].get(name, f"{name}_{li}.bin")
                mm, addr, size = map_file(os.path.join(root, fn), ro=False)   # L0 import needs a writable map
                assert self.E * rb <= size <= self.E * rb + 65536, (name, li, size, self.E * rb)   # tail padding allowed (CPU-tier overread guard)
                self.imp(addr, size)
                lst.append(addr); self.keep.append(mm)
        self.log(f"BT: mapped+imported {self.L} layers x 2 files in {time.time() - t0:.1f} s")
        self.scales = [scales_fn(li).to(self.dev) for li in range(self.L)]          # [E, NSC] fp16 per layer
        self.N = n_slots
        self.slots = torch.empty((n_slots, BLOB), dtype=torch.uint8, device=self.dev)
        self.slot_base = s64(self.slots.data_ptr())
        self.ptrs = torch.full((self.L, self.E), self.slot_base, dtype=torch.int64, device=self.dev)
        self.ptrs_cpu = np.full((self.L, self.E), self.slot_base, dtype=np.int64)
        self.lru = OrderedDict()          # (li, e) -> slot
        self.free = list(range(n_slots - 1, -1, -1))
        self.st13 = torch.empty((32, B13 // 2), dtype=torch.int16, device=self.dev)
        self.st2 = torch.empty((32, B2 // 2), dtype=torch.int16, device=self.dev)
        self.xd = torch.empty((MAXT, H), dtype=torch.float16, device=self.dev)
        self.od = torch.empty((MAXT, H), dtype=torch.float32, device=self.dev)
        self.stats = {"jobs": 0, "picks": 0, "staged": 0, "evicts": 0, "busy_s": 0.0}
        self.mb.res[:] = 0
        self.selfcheck_fn = None
        # kernel self-check (geometry/splits) before serving
        for M, k in ((1, 6), (4, 1)):
            ids = torch.zeros((M, k), dtype=torch.int32, device=self.dev); ww = torch.ones((M, k), device=self.dev)
            try:
                self.X.moe_forward(self.xd[:M], ids, ww, self.ptrs[0], I, K, self.E); torch.xpu.synchronize()
                self.log(f"BT: kernel check M={M} k={k} ok")
            except Exception as exc:
                self.log(f"BT: kernel check M={M} k={k} FAILED {exc!r} lib={os.environ.get('EXL3_MOE_LIB')} I={I} K={K} E={self.E}")
                raise

    # ---- residency
    def _admit(self, li, e, k):
        """Stage expert (li, e) through staging buffer k into a slot; returns slot."""
        if self.free:
            s = self.free.pop()
        else:
            (vl, ve), s = self.lru.popitem(last=False)
            self.mb.res[vl, ve] = 0; self.stats["evicts"] += 1
        X = self.X
        X.memcpy_async(s64(self.st13[k].data_ptr()), s64(self.h13[li] + e * B13), B13)
        X.memcpy_async(s64(self.st2[k].data_ptr()), s64(self.h2[li] + e * B2), B2)
        slot = self.slots[s]
        gu = self.st13[k].view(2, IT, OT, KW).permute(1, 0, 2, 3).reshape(IT, 2 * OT, KW)
        slot[:B13].view(torch.int32).copy_(planar4(gu).view(-1))
        slot[B13:B13 + B2].view(torch.int32).copy_(planar4(self.st2[k].view(OT, IT, KW)).view(-1))
        slot[B13 + B2:].view(torch.float16).copy_(self.scales[li][e])
        self.lru[(li, e)] = s
        self.ptrs_cpu[li, e] = self.slot_base + s * BLOB
        self.mb.res[li, e] = 1
        self.stats["staged"] += 1
        return s

    def ensure(self, li, experts):
        k = 0
        self._staged_now = 0
        changed = False
        for e in experts:
            key = (li, int(e))
            if key in self.lru:
                self.lru.move_to_end(key)
            else:
                self._admit(li, int(e), k % 32); k += 1; changed = True
                self._staged_now += 1
        if changed:
            self.ptrs[li].copy_(torch.from_numpy(self.ptrs_cpu[li]), non_blocking=False)

    # ---- one job
    def run_job(self, li, T, npk):
        p = self.mb.picks[:npk].copy()
        tok = p[:, 0].astype(np.int64); ex = p[:, 1].astype(np.int32); w = p[:, 2].view(np.float32)
        self.ensure(li, np.unique(ex))
        X = self.X
        X.memcpy_async(s64(self.xd.data_ptr()), s64(self.mb.base + self.mb.off_x), T * H * 2)
        if T == 1:
            ids = torch.from_numpy(ex.reshape(1, -1)).to(self.dev)
            ww = torch.from_numpy(w.reshape(1, -1).copy()).to(self.dev)
            y = X.moe_forward(self.xd[:1], ids, ww, self.ptrs[li], I, K, self.E)
            self.od[:1].copy_(y.float())
        else:
            # one launch over all T tokens: [T, kmax] picks, short rows padded with weight 0 on an id the token already
            # uses (L011: 7-18 % faster per job than gather -> M=npk,k=1 -> index_add, outputs equal within fp16 order)
            cnt = np.bincount(tok, minlength=T)
            km = int(cnt.max())
            ids_ = np.empty((T, km), np.int32); ww_ = np.zeros((T, km), np.float32)
            order = np.argsort(tok, kind="stable"); start = np.concatenate(([0], np.cumsum(cnt)[:-1]))
            for t in range(T):
                n_ = int(cnt[t]); sel = order[start[t]:start[t] + n_]
                if n_:
                    ids_[t, :n_] = ex[sel]; ww_[t, :n_] = w[sel]; ids_[t, n_:] = ex[sel[0]]
                else:
                    ids_[t, :] = ex[0]          # token without B70 picks: weight-0 row
            ids = torch.from_numpy(ids_).to(self.dev); ww = torch.from_numpy(ww_).to(self.dev)
            y = X.moe_forward(self.xd[:T], ids, ww, self.ptrs[li], I, K, self.E)
            self.od[:T].copy_(y.float())
        X.memcpy_async(s64(self.mb.base + self.mb.off_o), s64(self.od.data_ptr()), T * H * 4)
        torch.xpu.synchronize()
        self.stats["jobs"] += 1; self.stats["picks"] += npk

    def serve(self, stop=lambda: False, cpu=None):
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})
        c = self.mb.ctrl
        c[1] = c[0]; last = int(c[0])
        c[5] = int(time.time())
        c[6] = 1
        self.log(f"BT: serving, {self.N} slots ({self.N * BLOB / 2**30:.1f} GiB)")
        n = 0
        self.jl = []; self._staged_now = 0
        while not stop():
            s = int(c[0])
            if s == last:
                n += 1
                if n & 0xFFFF == 0:
                    c[5] = int(time.time())
                continue
            t = time.perf_counter()
            T_, np_ = int(c[3]), int(c[4])
            self.run_job(int(c[2]), T_, np_)
            c[1] = s; last = s
            dt = time.perf_counter() - t
            c[5] = int(time.time())
            self.stats["busy_s"] += dt
            J = self.jl; J.append((T_, np_, self._staged_now, dt * 1e3))
            if len(J) >= 20000:
                a = np.array(J); self.jl = []
                X_ = np.c_[np.ones(len(a)), a[:, 1] - a[:, 2], a[:, 2]]
                coef = np.linalg.lstsq(X_, a[:, 3], rcond=None)[0]
                q = np.percentile(a[:, 3], [50, 90, 99])
                self.log(f"BT: jobs {self.stats['jobs']} last20k: ms p50/p90/p99 {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}; fit ms = "
                         f"{coef[0]:.3f} + {coef[1]:.4f}*resident + {coef[2]:.3f}*staged; mean T {a[:, 0].mean():.2f} picks {a[:, 1].mean():.2f} "
                         f"staged {a[:, 2].mean():.2f}; evicts {self.stats['evicts']}")


def checkpoint_scales(model_dir, layer_names, E):
    """fp16 [E, NSC] per layer from the checkpoint: gate.suh, up.suh, gate.svh, up.svh, down.suh, down.svh."""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(n):
        f = idx[n]
        if f not in handles:
            handles[f] = safe_open(os.path.join(model_dir, f), "pt")
        return handles[f].get_tensor(n)

    def fn(li):
        L = layer_names[li]
        out = torch.empty((E, NSC), dtype=torch.float16)
        for e in range(E):
            b = f"layers.{L}.ffn.experts.{e}"
            parts = [get(f"{b}.w1.suh"), get(f"{b}.w3.suh"), get(f"{b}.w1.svh"), get(f"{b}.w3.svh"), get(f"{b}.w2.suh"), get(f"{b}.w2.svh")]
            out[e].copy_(torch.cat([p.half().reshape(-1) for p in parts]))
        return out
    return fn


def layout_check(w, model_dir, n=3, seed=0):
    """Blob built from the shared host store (server layout) == pack_expert(checkpoint) for n random (layer, expert)."""
    from safetensors import safe_open
    from exl3xpu.moe_offload import pack_expert
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    rng = np.random.default_rng(seed); bad = 0
    for _ in range(n):
        li = int(rng.integers(w.L)); e = int(rng.integers(w.E)); L = w.meta["layers"][li]["ckpt_layer"]
        hd = {}
        def get(nm):
            f = idx[nm]
            if f not in hd: hd[f] = safe_open(os.path.join(model_dir, f), "pt")
            return hd[f].get_tensor(nm)
        ws = [{s_: get(f"layers.{L}.ffn.experts.{e}.{nm}.{s_}") for s_ in ("trellis", "suh", "svh")} for nm in ("w1", "w3", "w2")]
        ref = pack_expert(*ws, 3)
        w.ensure(li, [e]); torch.xpu.synchronize()
        got = w.slots[w.lru[(li, e)]].cpu()
        ok = bool(torch.equal(got, ref)); bad += (not ok)
        w.log(f"BT: layout check layer {li} (ckpt {L}) expert {e}: {'OK' if ok else 'MISMATCH'}")
    return bad == 0


if __name__ == "__main__":
    os.environ["EXL3_MOE_LIB"] = os.environ.get("BT_MOE_LIB", "/out/_moe-dsv41.so")
    from exl3xpu.moe_offload import ops
    root, model, n_slots = sys.argv[1], sys.argv[2], int(sys.argv[3])
    meta = json.load(open(os.path.join(root, "meta.json")))
    t0 = time.time()
    w = Worker(root, n_slots, checkpoint_scales(model, [l["ckpt_layer"] for l in meta["layers"]], int(meta["E"])), ops())
    print(f"BT: init {time.time() - t0:.1f} s", flush=True)
    if not layout_check(w, model):
        print("BT: layout check FAILED, not serving", flush=True); sys.exit(3)
    w.serve(cpu=int(os.environ.get("BT_CPU", "25")))
