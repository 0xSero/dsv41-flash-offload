#!/bin/bash
set -euo pipefail
cd ~/freetoken-exl3
out=runs/2026-10-02-D061-prefill16384-dma-gemm-fat32; name=dsv41-D061
trap 'timeout 30 docker stop -t 15 $name >/dev/null 2>&1 || true' EXIT
awk '/MemAvailable/{print}' /proc/meminfo > $out/memory-before.txt
timeout 60 docker rm -f $name >/dev/null 2>&1 || true
timeout 120 docker run -d --name $name --gpus device=0 --network host --memory=300g --memory-swap=300g --cpus=24 --shm-size=8g --cap-add IPC_LOCK --ulimit memlock=-1:-1 -e PYTHONPATH=/out/runtime-source:/out/cpu-base:/w -e OMP_NUM_THREADS=16 -e DSV41_PACK=/models/DSV41-EXL3-3090-D010 -e DSV41_ENGRAM_DISK=1 -e EXL3_HOST_EXPERTS=/models/DSV41-EXL3-3090-D010/host-plan.json -e VLLM_EXL3_MOE_KERNEL=exllamav3 -e VLLM_EXL3_FAT_THRESHOLD=32 -e EXL3_DMA_GEMM=1 -e EXL3_HOST_DMA=1 -e EXL3_HOST_DMA_BATCH=8 -e EXL3_HOST_DMA_SLOTS=2 -e EXL3_HOST_DMA_COLD_FUSED=1 -e DSV41_EC=0 -e VLLM_PLUGINS=vllm_exl3 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e TORCH_CUDA_ARCH_LIST=8.6 -e FLASHINFER_CUDA_ARCH_LIST=8.6 -e DSV41_CPU_TIER=1 -v $HOME/models:/models:ro -v $PWD:/w:ro -v $PWD/$out:/out -v $PWD/dsv41/cache:/root/.cache --entrypoint bash dsv41-exl3:ampere-d005 /out/inner.sh >/dev/null
( timeout 43200 docker logs -f $name > $out/server.log 2>&1 & )
for i in $(seq 1 480); do
  curl -sf -m 5 http://127.0.0.1:30141/v1/models >/dev/null && break
  timeout 10 docker inspect -f '{{.State.Running}}' $name 2>/dev/null | grep -q true || { echo "server died"; exit 3; }
  sleep 5
done
curl -sf -m 5 http://127.0.0.1:30141/v1/models >/dev/null || { echo "server not ready"; exit 4; }
echo "ready $(date -Is)"; nvidia-smi --query-gpu=memory.used --format=csv,noheader > $out/vram.txt
awk '/MemAvailable/{print}' /proc/meminfo > $out/memory-ready.txt
python3 -u dsv41/resources.py --container "$name" --out "$out/resources.jsonl" > "$out/resources.log" 2>&1 &
timeout 7200 python3 "$out/runtime-source/sweep_prefill_terminal.py" --api vllm --template deepseek --url http://127.0.0.1:30141 --card rtx3090 --config "D061-CPU1-KV1-noDSpark-chunk16384-dma-gemm-fat32" --prefill 8192 32768 --conc 1 2 4 --reps 3 --dec-reps 2 --vocab-max 128000 --no-early-exit --prefill-only --out $out/sweep.json > $out/sweep.log 2>&1
awk '/MemAvailable/{print}' /proc/meminfo > $out/memory-after.txt
grep -a "dsv41 cpu_tier" $out/server.log | tail -5

rg -q "EXL3 stock DMA GEMM ACTIVE" "$out/server.log" || { echo "BLOCKED: DMA did not activate" > "$out/result.txt"; exit 5; }
mkdir -p "$out/prefill"
timeout 10800 docker exec "$name" python3 -u /w/dsv41/ref_panel.py --ref /w/runs/2026-10-01-D026-kv2-reference-p2/reference --out /out/prefill > "$out/prefill.log" 2>&1
python3 - "$out" <<'RECEIPT'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'prefill/result.json').read_text());passed=r['top1_agreement']>=.988 and r['mean_kl_nats']<=.00103
r['inherited_guard_passed']=passed;(p/'fidelity-verdict.json').write_text(json.dumps(r,indent=2)+'\n')
(p/'result.txt').write_text(f"COMPLETE diagnostic prefill screen; GPU parity passed; inherited prefill fidelity {'PASS' if passed else 'FAIL'}; full P2 required before promotion\n")
RECEIPT

# Profiling is separate and never included in the prefill speed medians.
if timeout 3600 docker exec "$name" python3 /out/runtime-source/profile_prefill.py --out /out/model-profile-run.json > "$out/model-profile.log" 2>&1; then
    echo "diagnostic model profile complete"
else
    echo "diagnostic model profile unavailable: $?" > "$out/model-profile-unavailable.txt"
fi
