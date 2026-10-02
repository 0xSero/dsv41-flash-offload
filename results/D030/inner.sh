#!/bin/bash
set -euo pipefail
python3 /w/dsv41/patch_runtime.py > /out/runtime-patch-receipt.json
exec vllm serve "$DSV41_PACK" --host 127.0.0.1 --port 30141 --tensor-parallel-size 1 --quantization exl3 --served-model-name deepseek-v4.1-flash --max-logprobs -1 --max-model-len 65536 --max-num-seqs 4 --max-num-batched-tokens 1024 --kv-cache-dtype fp8 --kv-cache-memory 2147483648 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --language-model-only --tokenizer-mode deepseek_v41 --trust-remote-code --disable-custom-all-reduce --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}' --cudagraph-capture-sizes 1 2 4 6 8 12 16 24
