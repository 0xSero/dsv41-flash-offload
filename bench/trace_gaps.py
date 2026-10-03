# Parse a torch profiler trace: GPU timeline of HtoD memcpy vs compute kernels; busy unions, overlap, idle gaps.
import gzip, json, sys
t = json.load(gzip.open(sys.argv[1]))
ev = [e for e in t["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("gpu_memcpy", "kernel")]
cp = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e["cat"] == "gpu_memcpy" and "HtoD" in e["name"])
kn = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e["cat"] == "kernel")
def union(iv):
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
U_cp, U_kn = union(cp), union(kn)
t0 = min(cp[0][0], kn[0][0]); t1 = max(cp[-1][1], kn[-1][1])
tot = lambda u: sum(b - a for a, b in u)
def inter(u, v):
    i = j = s = 0
    while i < len(u) and j < len(v):
        a, b = max(u[i][0], v[j][0]), min(u[i][1], v[j][1])
        if a < b: s += b - a
        if u[i][1] < v[j][1]: i += 1
        else: j += 1
    return s
both = inter(U_cp, U_kn); any_ = tot(union(sorted(cp + kn)))
print(f"window {(t1-t0)/1e6:.2f} s; copy busy {tot(U_cp)/1e6:.2f} s; kernel busy {tot(U_kn)/1e6:.2f} s; overlap {both/1e6:.2f} s; GPU idle (neither) {((t1-t0)-any_)/1e6:.2f} s")
# biggest idle gaps (neither copy nor kernel)
au = union(sorted(cp + kn)); gaps = sorted(((au[i+1][0] - au[i][1]), au[i][1]) for i in range(len(au) - 1))[::-1][:8]
print("largest all-idle gaps ms:", [round(g / 1e3, 1) for g, _ in gaps])
# copy-idle time while kernels run, i.e. potential overlap left
print(f"kernel-only {(tot(U_kn)-both)/1e6:.2f} s; copy-only {(tot(U_cp)-both)/1e6:.2f} s")
