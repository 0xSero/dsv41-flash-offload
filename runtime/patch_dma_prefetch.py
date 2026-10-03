#!/usr/bin/env python3
"""D135/D139: cross-layer prefetch (and D139: skip the CPU-tier tail batches of a hybrid step) on the stock DMA prefill path (applied after patch_dma_gemm.py). The stock staged loop issued
its first host->VRAM copies only when a MoE layer started, so the copy engine idled through every attention block
(profile: ~3.3 s of kernel-only time per 8k step). After a layer's last batch, queue the next MoE layer's first batches;
each copy waits for the last reader of its slot. Prefill 8k/64k/261k ~710/700/620 -> 787/769/694 tok/s, outputs identical.
EXL3_DMA_STOCK_PREFETCH=0 turns it off."""
import hashlib, json, sys
from pathlib import Path
if len(sys.argv) > 1:
    p = Path(sys.argv[1])
else:
    import vllm_exl3
    p = Path(vllm_exl3.__file__).parent/'exl3.py'
s = p.read_text()
before = hashlib.sha256(s.encode()).hexdigest()
CHANGES = [
    ('_DMA_STATE: dict = {}',
     '_DMA_STOCK_PREFETCH = os.environ.get("EXL3_DMA_STOCK_PREFETCH", "1") == "1"   # dsv41: cross-layer prefetch on the stock DMA path\n'
     '_DMA_STATE: dict = {}'),
    ('        nb = len(D["batches"])\n        events = {j: _dma_issue(layer, j, dev) for j in range(min(_DMA_NS, nb))}\n',
     '        nb = len(D["batches"]) - int(getattr(layer, "_dsv41_cpu_tail", 0) or 0)   # dsv41 hybrid: the CPU tier computes the last batches\n'
     '        events = {j: _dma_issue(layer, j, dev) for j in range(min(_DMA_NS, nb))}\n'),
    ('        events = {j: _dma_issue(layer, j, dev) for j in range(min(_DMA_NS, nb))}\n',
     '        pre = D.pop("pre", None) or {}   # dsv41: first batches prefetched by the previous MoE layer (copied during attention)\n'
     '        D["pre"] = {}\n'
     '        events = {j: (pre[j] if j in pre else _dma_issue(layer, j, dev)) for j in range(min(_DMA_NS, nb))}\n'),
    ('                events[j + _DMA_NS] = _dma_issue(layer, j + _DMA_NS, dev)\n'
     '        logger.info_once("EXL3 stock DMA prefill ACTIVE',
     '                events[j + _DMA_NS] = _dma_issue(layer, j + _DMA_NS, dev)\n'
     '        nxt = D.get("next")\n'
     '        if _DMA_STOCK_PREFETCH and nxt is not None and getattr(nxt, "_exl3_dma", None) is not None:\n'
     "            # dsv41: queue the next MoE layer's first batches now; each copy waits for this layer's last reader of its slot,\n"
     "            # then runs on the copy engine while the next layer's attention computes (stock path had no cross-layer prefetch)\n"
     '            nd = nxt._exl3_dma\n'
     '            nd["pre"] = {j: _dma_issue(nxt, j, dev) for j in range(min(_DMA_NS, len(nd["batches"])))}\n'
     '        logger.info_once("EXL3 stock DMA prefill ACTIVE'),
]
for old, new in CHANGES:
    assert s.count(old) == 1, (old[:60], s.count(old))
    s = s.replace(old, new)
compile(s, str(p), 'exec')
p.write_text(s)
print(json.dumps({'file': str(p), 'before': before, 'after': hashlib.sha256(s.encode()).hexdigest(), 'strategy': 'stock DMA path: next-layer prefetch'}))
