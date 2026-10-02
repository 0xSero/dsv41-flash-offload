#!/usr/bin/env python3
"""Add staged prefill for the pinned stock group kernel, without negative counts."""
import hashlib, json
from pathlib import Path
import vllm_exl3
p = Path(vllm_exl3.__file__).parent/'exl3.py'
s = p.read_text()
before = hashlib.sha256(s.encode()).hexdigest()
old = '''            finally:
                force(0)
    elif n_active_host is not None:
        fn(*args, n_active_host)
'''
new = '''            finally:
                force(0)
    elif dma_on:
        # Stock EXL3 uses group kernels but lacks thor split/lock hooks and
        # interprets negative counts as negative offsets. Repack routes per
        # launch so all counts and offsets follow the stock contract.
        assert _EXL3_HOST_DMA_MODE == 1, "Stock DMA fallback supports mode 1"
        dev = x2d.device
        D = _dma_tables(layer, dev, n_exp)
        st = _DMA_STATE[dev.index]
        main = torch.cuda.current_stream(dev)
        def launch(mask, staged):
            safe = standard_local.clamp(max=n_exp)
            selected = mask.index_select(0, safe) & (standard_local < n_exp)
            batch_local = standard_local.masked_fill(~selected, n_exp)
            order = batch_local.argsort()
            count = torch.zeros(n_exp + 1, dtype=torch.long, device=dev)
            count.scatter_add_(0, batch_local.long(), torch.ones_like(batch_local, dtype=torch.long))
            la = list(args)
            la[2:5] = [count, flat_token[order], standard_weight[order]]
            if staged:
                ck = int(getattr(layer, "_exl3_cold_k", k))
                la[10:13] = [ck, ck, ck]
                la[13] = D["ptr"]["gate_trellis"]
                la[16] = D["ptr"]["up_trellis"]
                la[19] = D["ptr"]["down_trellis"]
            fn(*la, n_active_host) if n_active_host is not None else fn(*la)
        if D["n_cold"] < n_exp:
            launch(~cold_mask, False)
        nb = len(D["batches"])
        events = {j: _dma_issue(layer, j, dev) for j in range(min(_DMA_NS, nb))}
        for j in range(nb):
            main.wait_event(events[j])
            launch(D["bmask"][j], True)
            done = torch.cuda.Event()
            done.record(main)
            st["free"][j % _DMA_NS] = done
            if j + _DMA_NS < nb:
                events[j + _DMA_NS] = _dma_issue(layer, j + _DMA_NS, dev)
        logger.info_once("EXL3 stock DMA prefill ACTIVE: positive-count route batches, %d experts x %d slots", _DMA_B, _DMA_NS)
    elif n_active_host is not None:
        fn(*args, n_active_host)
'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
compile(s, str(p), 'exec')
p.write_text(s)
print(json.dumps({'file':str(p), 'before':before, 'after':hashlib.sha256(s.encode()).hexdigest(), 'strategy':'stock group kernel, repacked positive-count routes, stream-ordered staging'}))
