#!/usr/bin/env python3
"""Keep cold weights staged while splitting oversized expert route lists.

Apply only after patch_dma_stock.py. Prefill-only, mode1/cold-fused, stock ABI.
"""
import hashlib,json
from pathlib import Path


def patch_source(src):
    edits = [
        ('    slice_ok = not (_EXL3_NO_FAT_SLICE and 0 < FAT_EXPERT_THRESHOLD < TEMP_ROWS_FUSED)', '''    stock_dma_segmented = (
        _EXL3_HOST_DMA_MODE == 1 and _EXL3_DMA_COLD_FUSED
        and tokens >= _DMA_MIN_TOKENS and tokens > 64
        and getattr(layer, "_exl3_cold_mask", None) is not None
        and getattr(layer, "_exl3_dma", None) is not None
        and getattr(exllamav3_ext, "xmoe_set_force_group", None) is None
        and not torch.cuda.is_current_stream_capturing()
    )
    slice_ok = not stock_dma_segmented and not (_EXL3_NO_FAT_SLICE and 0 < FAT_EXPERT_THRESHOLD < TEMP_ROWS_FUSED)'''),
        ('            dma_on = max((counts_host_dma[e] for e in layer._exl3_dma["h13"].cold), default=0) <= TEMP_ROWS_FUSED', '            dma_on = stock_dma_segmented or max((counts_host_dma[e] for e in layer._exl3_dma["h13"].cold), default=0) <= TEMP_ROWS_FUSED'),
        ('''            fn(*la, n_active_host) if n_active_host is not None else fn(*la)
        if D["n_cold"] < n_exp:''', '''            if staged and stock_dma_segmented:
                # order groups the rows of each selected expert. Keep that
                # expert's weights in its slot until every row pass finishes.
                sorted_local = batch_local[order]
                start = count.cumsum(0) - count
                rank = torch.arange(sorted_local.numel(), device=dev) - start[sorted_local]
                passes = (int(count[:n_exp].max().item()) + TEMP_ROWS_FUSED - 1) // TEMP_ROWS_FUSED
                for p in range(passes):
                    keep = ((sorted_local < n_exp) & (rank >= p * TEMP_ROWS_FUSED)
                            & (rank < (p + 1) * TEMP_ROWS_FUSED)).nonzero(as_tuple=True)[0]
                    segment_count = torch.zeros_like(count)
                    segment_count.scatter_add_(0, sorted_local[keep].long(), torch.ones_like(keep, dtype=torch.long))
                    la[2:5] = [segment_count, flat_token[order][keep], standard_weight[order][keep]]
                    fn(*la, n_active_host) if n_active_host is not None else fn(*la)
            else:
                fn(*la, n_active_host) if n_active_host is not None else fn(*la)
        if D["n_cold"] < n_exp:'''),
        ('"EXL3 stock DMA prefill ACTIVE: positive-count route batches, %d experts x %d slots"', '"EXL3 stock DMA prefill ACTIVE: positive-count expert segments, %d experts x %d slots"'),
    ]
    for old,new in edits:
        assert src.count(old)==1,(old[:70],src.count(old))
        src=src.replace(old,new)
    compile(src,'segmented-stock-dma-exl3.py','exec')
    return src


if __name__=='__main__':
    import vllm_exl3
    p=Path(vllm_exl3.__file__).parent/'exl3.py'
    src=p.read_text();patched=patch_source(src);p.write_text(patched)
    print(json.dumps(dict(file=str(p),before=hashlib.sha256(src.encode()).hexdigest(),after=hashlib.sha256(patched.encode()).hexdigest(),strategy='one weight copy per batch; <=2048 routed rows/expert/pass')))
