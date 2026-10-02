#!/bin/bash
set -euo pipefail
python3 /w/dsv41/patch_runtime.py > /out/runtime-patch-receipt.json
P=$(python3 -c "import vllm_exl3,os;print(os.path.dirname(vllm_exl3.__file__))")
grep -q dsv41.ct.ct_vllm $P/exl3.py || printf '\ntry:\n    import sys as _s, dsv41.ct.ct_vllm as _dsct\n    _dsct.install(_s.modules[__name__])\nexcept Exception as _e:\n    print("dsv41 ct install failed", repr(_e), flush=True)\n' >> $P/exl3.py
SPEC=()
[ "none" != none ] && SPEC=(--speculative-config 'none')
exec vllm serve "$DSV41_PACK" --host 127.0.0.1 --port 30141 --tensor-parallel-size 1 --quantization exl3 --served-model-name deepseek-v4.1-flash --max-model-len 65536 --max-num-seqs 4 --max-num-batched-tokens 1024 --kv-cache-dtype fp8 --kv-cache-memory 1073741824 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --language-model-only --tokenizer-mode deepseek_v41 --trust-remote-code --disable-custom-all-reduce "${SPEC[@]}" --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}' --cudagraph-capture-sizes 1 2 4 6 8 12 16 24 
