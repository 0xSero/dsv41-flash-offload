#!/usr/bin/env python3
"""Optional EXL3_DMA_GEMM=1 cold prefill on staged weights; stock+segments first."""
from pathlib import Path
import hashlib,json


def patch_source(src):
    edits=[
        ('    slice_ok = not stock_dma_segmented and not', '    stock_dma_gemm = stock_dma_segmented and os.environ.get("EXL3_DMA_GEMM", "0") == "1"\n    slice_ok = not stock_dma_segmented and not'),
        ('        def launch(mask, staged):', '        gemm_order = local.argsort() if stock_dma_gemm else None\n        def launch(mask, staged, batch=None):'),
        ('            safe = standard_local.clamp(max=n_exp)', '''            if staged and stock_dma_gemm:
                # Busy cold routes run exactly once via the existing fat GEMM
                # from this same slot; the group launch handles only small ones.
                mask = mask & torch.cat((counts <= FAT_EXPERT_THRESHOLD, torch.zeros(1, dtype=torch.bool, device=dev)))
            safe = standard_local.clamp(max=n_exp)'''),
        ('            if staged and stock_dma_segmented:', '            if staged and stock_dma_segmented and not stock_dma_gemm:'),
        ('        if D["n_cold"] < n_exp:\n            launch(~cold_mask, False)', '''            if staged and stock_dma_gemm:
                from dma_gemm import run_batch
                run_batch(D, st, batch, inners, counts_host_dma, FAT_EXPERT_THRESHOLD,
                          xh, flat_token[gemm_order], flat_weight[gemm_order], limit, out,
                          apply_exl3_batched_fat, _fat_kernel_available())
                logger.info_once("EXL3 stock DMA GEMM ACTIVE: busy experts reconstruct from staging slots; small experts positive-count fused")
        if D["n_cold"] < n_exp:
            launch(~cold_mask, False)'''),
        ('            launch(D["bmask"][j], True)', '            launch(D["bmask"][j], True, j)'),
    ]
    for old,new in edits:
        assert src.count(old)==1,(old[:80],src.count(old))
        src=src.replace(old,new)
    compile(src,'dma-gemm-exl3.py','exec')
    return src


if __name__=='__main__':
    import vllm_exl3
    p=Path(vllm_exl3.__file__).parent/'exl3.py';old=p.read_text();new=patch_source(old);p.write_text(new)
    print(json.dumps({'file':str(p),'before':hashlib.sha256(old.encode()).hexdigest(),'after':hashlib.sha256(new.encode()).hexdigest(),'gate':'EXL3_DMA_GEMM=1; stock segmented DMA prefill only','strategy':'stage once; existing reconstruct/GEMM for busy cold routes; positive group counts for small routes'}))
