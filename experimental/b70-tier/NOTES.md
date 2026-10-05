# 16. Decode scaling with concurrency: physics bound, and a B70 expert tier (HOM-238, 2026-10-04)

## Measured baseline (D141, live, steady-state decode window, no output caps)

| C | aggregate tok/s | step ms |
|---|---|---|
| 1 | 19.3 | 51.8 |
| 2 | 24.0 | 83 |
| 3 | 22.4 | 134 |
| 4 | 26.6 | 150 |

Torch profile at C4:
- Routed MoE kernel (EC hits plus PCIe zero-copy misses): 2.65 ms per layer, at about 24.5 GB/s, i.e. link-bound.
- CPU-tier wait: 0.36 ms per layer.
- Fixed GPU work: 12 ms per step, about 8% of the step.

Why it saturates: each extra stream adds about 6 distinct experts per layer (sharing is only 2% at C4). The two expert engines are both saturated: the CPU tier at about 5.4 experts/ms and PCIe at about 1.85 experts/ms. Linear scaling would need about 60% of picks served from fast memory at C4, which is 24 GB of VRAM against 204 GB of experts.

## Real DSV41 routing (cand10 capture ring)

- 25,000 decode tokens, 16 diverse prompts. The EC reached a hit rate of 0.38-0.44 with only 392 slots.
- Calibrated simulator (`sim_bt.py`): step = 1.184 x (sum of per-layer routed time) + 19.4 ms. It fits the measured C1/C2/C4 times.

## B70 expert tier (cand11: dsv41/ct/cand11, dsv41/ct/bt/bt_worker.py, lab_run_cand11.sh)

Design:
- The host store lives in shared tmpfs files. The 3090 server registers them with cudaHostRegister, and the B70 worker imports them through Level Zero (`zexDriverImportExternalPointer`, DMA at 26-28 GB/s, bit-exact).
- The worker uses LRU slots with write-through staging: DMA, then device PLANAR4 conversion (0.04 ms per expert), then compute.
- ft_split uses a 3-engine cost model: the CPU takes the coldest experts, the 3090 zero-copies the middle, and the B70 takes its resident experts plus the hottest misses.

Measured on the B70:
- Resident expert compute 0.031-0.048 ms per expert.
- Python worker job: 0.42 ms at T1 with 6 picks, 1.0 ms at T4 with 24 picks.

Live cand11 results:

| C | D141 tok/s | cand11 tok/s | change |
|---|---|---|---|
| 1 | 19.3 | 24.7 | +28% |
| 2 | 24.0 | 34.6 | +44% |
| 3 | 22.4 | 35.0 | +56% |
| 4 | 26.6 | 41.3 | +55% |

- C4/C1 scaling went from 1.38x to 1.67x.
- The B70 held 35% of picks; there were 0 B70 timeouts.
- The sim had projected 27/41/55. The C4 gap is the CPU tier being the long pole: ft_combine waits 0.67 ms per layer, so the cost-model constants need recalibrating.

**Blocker:** cand11 segfaulted twice in the CPU-tier kernel (`gemv_block_i16`), about once per 15-20 min of decode. The cause is not identified yet. Suspects:
1. Shared tmpfs store plus MADV_HUGEPAGE / khugepaged on kernel 7.2 with CUDA-pinned and Level Zero-imported pages. THP is now opt-in via `EXL3_HOST_SHM_HUGE`.
2. A CPU-tier job race that the new pick mix exposes.

The k2b extension copy adds a SIGSEGV/SIGBUS reporter (address, job, store bounds). The server is back on D141.

## Next steps

1. Debug run of cand11 (no shmem THP, k2b reporter). Needs the user's go-ahead for more live switches.
2. Recalibrate the cost model from worker and CPU job timings.
3. Port the worker to C++ (about 0.15 ms of Python overhead per job).
4. A second B70 (84:00.0) would roughly double the tier (sim: C4 about 60+).

## Segfault root cause and fix (2026-10-05)

**Repro.** The harness `dsv41/ct/rep/` (`rep_run.sh` and `rep_ct.py`) reproduces the cand11 setup without GPU0:
- 4 real layers in `/dev/shm/dsv41rep`.
- The k2b CPU tier with its SIGSEGV reporter.
- The real B70 worker on c3:00.0 with 300 slots.
- A replay of captured routing plus hybrid 513-2048-token jobs.
- A watcher that pauses the harness whenever GPU0 is busy.

Results:
- **REP-huge1** (THP on, no pad): crashed in under 1 minute, about 5.4k jobs. The report was `signal 11 code 1 (MAPERR) addr 0x7f87d6000000`, which is exactly `store_hi`.
- **REP-nopad-huge0** (THP off, no pad): the same crash at the store end, about 6.1k jobs. THP is ruled out.

**Cause.** `ft_core.h decode8p<g>` loads 16 bytes at `tile + 4*((3g-2)/4)`. For g=31 that is `tile+88`, which reads bytes 88-103 of a 96-byte tile, 8 bytes past its end.
- Each cand11 store file is mapped at exactly its size, `torch.from_file(size=n)`, and the size is a page multiple.
- So the final tile of expert 383 sits flush against an unmapped page.
- D141's store is a large `torch.empty`. glibc serves it with an mmap that carries a 16-byte header, so the data never ends on a page boundary and the overread is harmless.

**Fix.**
- `cand11/exl3bt/exl3.py` now creates every shared store file 4 KiB larger and maps `n + 2048` int16, then slices `[:n]`.
- `bt_worker.py` accepts padded files.

**Validation.** REP-pad4k (pad 4096, 40 min): no crash. The run covered 1.76M steps, 1.53M CPU-tier jobs, 4.59M B70 picks, 293 hybrid jobs and 2.75M B70 evictions.

## B70 worker multi-token path (L011, 2026-10-05)

The multi-token path used to gather rows, launch with M = npk and k = 1, then scatter with index_add. It is now a single launch over [T, kmax], with short rows padded at weight 0.

Per-job p50 (ms), old -> new:

| Job shape | Old | New |
|---|---|---|
| T1x2 | 0.253 | 0.207 |
| T2x6 | 0.436 | 0.366 |
| T4x8 | 0.486 | 0.423 |
| T4x12 | 0.611 | 0.520 |
| T4x16 | 0.732 | 0.665 |
| T8x24 | 0.951 | 0.910 |

Through the real run_job, 60 random jobs matched the old path within 3.0e-4 relative (fp16 order), and rows without picks stay exactly 0.

The C4 profile (L010, cand11, about 91 ms per step) is per layer:
- 3090 routed kernel: 0.89 ms
- ft_combine wait: 0.67 ms

So the CPU tier and the B70 both run about 1.55 ms per layer while the 3090 sits idle. The cost model's constants were calibrated at C1 and under-predict CPU and B70 cost at C4. Re-tuning them with live cand11 runs is the next lever; a balanced 3-way split is worth about +7% at C4.

## Quality status

So far only kernel numerics, 2 smoke prompts and speed windows have been checked. Long-prompt teacher-forced comparison against D141 (KLD and top-1), long tasks and a soak are pending live cand11 time.
