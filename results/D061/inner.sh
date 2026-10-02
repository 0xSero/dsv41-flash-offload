#!/bin/bash
set -euo pipefail
python3 /out/runtime-source/patch_runtime.py > /out/runtime-patch-receipt.json
python3 /out/runtime-source/patch_dma_stock.py > /out/dma-patch-receipt.json
python3 /out/runtime-source/patch_dma_segments.py > /out/dma-segments-receipt.json
python3 /out/runtime-source/patch_dma_gemm.py > /out/dma-gemm-receipt.json
timeout 1800 python3 /out/runtime-source/probe_dma.py --profile --segmented --gemm --out /out/dma-kernel-parity.json > /out/dma-kernel-parity.log 2>&1
P=$(python3 -c "import vllm_exl3,os;print(os.path.dirname(vllm_exl3.__file__))")
grep -q dsv41.ct.ct_vllm $P/exl3.py || printf '\ntry:\n    import sys as _s, ct.ct_vllm as _dsct\n    _dsct.install(_s.modules[__name__])\nexcept Exception as _e:\n    print("dsv41 ct install failed", repr(_e), flush=True)\n' >> $P/exl3.py
exec vllm serve "$DSV41_PACK" --host 127.0.0.1 --port 30141 --tensor-parallel-size 1 --quantization exl3 --served-model-name deepseek-v4.1-flash --max-logprobs -1 --max-model-len 65536 --max-num-seqs 4 --max-num-batched-tokens 16384 --kv-cache-dtype fp8 --kv-cache-memory 1073741824 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --language-model-only --tokenizer-mode deepseek_v41 --trust-remote-code --disable-custom-all-reduce --profiler-config '{"profiler":"torch","torch_profiler_dir":"/out/model-profile","torch_profiler_with_stack":false,"ignore_frontend":true,"max_iterations":1}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}' --cudagraph-capture-sizes 1 2 4 6 8 12 16 24 
