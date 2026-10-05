# Experimental: Intel Arc B70 expert tier (cand11)

Status: **experimental, not the default.** The live default is still D141 (see the top-level README).

## Measured on omarchy

Hardware: 1x RTX 3090 and 1x Arc B70 at PCIe c3:00.0.

| concurrent streams | D141 tok/s | cand11 tok/s | change |
|---|---|---|---|
| 1 | 19.3 | 24.7 | +28% |
| 2 | 24.0 | 34.6 | +44% |
| 3 | 22.4 | 35.0 | +56% |
| 4 | 26.6 | 41.3 | +55% |

All numbers are steady-state decode with no output caps. Going from 1 to 4 streams multiplies throughput by 1.67 (D141: 1.38).

## How it works

- **Shared store.** The routed-expert host store moves into shared tmpfs files (`ct/exl3-shared-store.patch`). The 3090 server registers them with `cudaHostRegister`.
- **B70 worker.** A separate XPU worker (`worker/bt_worker.py`) maps the same files and imports them with Level Zero (`zexDriverImportExternalPointer`). It DMAs experts at 26-28 GB/s with no second copy in RAM.
  - It keeps about 2,200 experts (27 GB) resident in LRU slots.
  - Admission is write-through: DMA, then PLANAR4 conversion, then compute.
- **Three-way split.** The split kernel (`ct/ft_tier_cu_v.cu`) picks an engine per expert:
  - 3090 VRAM cache, or 3090 zero-copy over PCIe
  - the AVX2 CPU tier
  - the B70: its resident experts, plus the hottest misses, which get staged
- **Mailbox.** Jobs reach the B70 through a shared mailbox file. If the worker stalls, the server turns the B70 tier off and keeps serving.

## Crash found and fixed (2026-10-05)

The first live runs segfaulted every 15-20 minutes in the CPU tier (`gemv_block_i16`).

- **Cause.** The AVX2 tile decoder (`decode8p`, group 31) loads 16 bytes at `tile + 88`. That reads 8 bytes past each 96-byte tile.
  - The shared-store files were mapped at exactly their size, so the last tile of the last expert touched an unmapped page.
  - D141 is immune: its store comes from glibc mmap, whose 16-byte header means the data never ends on a page boundary.
- **Repro without the 3090** (`repro/`: real experts, the real B70 worker, captured routing replay):
  - Without the pad it crashes within about 1 minute, with THP both on and off. The faulting address is exactly the store end.
- **Fix.** A 4 KiB tail pad on every store file.
  - With the pad the repro ran 40 minutes clean: 1.76M decode steps, 1.53M CPU-tier jobs, 4.59M B70 picks, 293 hybrid jobs.

## Other results

- **One padded launch for multi-token B70 jobs** (`bench/b70_tpath.py`, `results/L011-b70-tpath.json`): 7-18% faster per job.
  - Outputs match the old gather/scatter path within 3e-4 relative (`bench/b70_jobcheck.py`, 60 random jobs).
- **SMT siblings for the CPU tier** (`results/L012-smt.jsonl`): rejected.
  - p50 gains are at most about 13%, but p90 tails are 3-6x worse, and the GPU waits on the slowest job.
- **Next lever.** At 4 streams the 3090 idles 0.67 ms per layer while the CPU tier and the B70 each take about 1.55 ms.
  - Re-tuning the cost model is worth roughly +7%.

## Before it can become the default

1. Long-prompt quality A/B against D141 with `bench/qa_long.py`. The D141 reference is in `results/Q001-d141-quality/`.
2. A Pi agent session.
3. A 60-minute soak on the live server.

The scripts here are lab scripts. They assume the campaign's `~/freetoken-exl3` layout and the omarchy device map. Full notes are in [NOTES.md](NOTES.md).
