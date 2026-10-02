#!/bin/bash
set -uo pipefail
cd ~/freetoken-exl3
out=runs/2026-10-02-D107-ec-kv1g-margin500; name=dsv41-D107
trap 'timeout 30 docker stop -t 15 $name >/dev/null 2>&1 || true' EXIT
awk '/MemAvailable/{print}' /proc/meminfo > $out/memory-before.txt
timeout 60 docker rm -f $name >/dev/null 2>&1 || true
timeout 120 docker run -d --name $name --gpus device=0 --network host --memory=300g --memory-swap=300g --cpus=24 --shm-size=8g --cap-add IPC_LOCK --ulimit memlock=-1:-1 -e PYTHONPATH=/w -e OMP_NUM_THREADS=16 -e DSV41_PACK=/models/DSV41-EXL3-3090-D010 -e DSV41_ENGRAM_DISK=1 -e EXL3_HOST_EXPERTS=/models/DSV41-EXL3-3090-D010/host-plan.json -e EXL3_MOE_KERNEL=exllamav3 -e VLLM_PLUGINS=vllm_exl3 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e TORCH_CUDA_ARCH_LIST=8.6 -e FLASHINFER_CUDA_ARCH_LIST=8.6 -e DSV41_CPU_TIER=1 -e EXL3_HOST_DMA=2 -e DSV41_EC=1 -e DSV41_EC_MARGIN_MB=500 -v $HOME/models:/models:ro -v $PWD:/w:ro -v $PWD/$out:/out -v $PWD/dsv41/cache:/root/.cache --entrypoint bash dsv41-exl3:ampere-d005 /out/inner.sh >/dev/null
( timeout 43200 docker logs -f $name > $out/server.log 2>&1 & )
for i in $(seq 1 480); do
  curl -sf -m 5 http://127.0.0.1:30141/v1/models >/dev/null && break
  timeout 10 docker inspect -f '{{.State.Running}}' $name 2>/dev/null | grep -q true || { echo "server died"; exit 3; }
  sleep 5
done
curl -sf -m 5 http://127.0.0.1:30141/v1/models >/dev/null || { echo "server not ready"; exit 4; }
echo "ready $(date -Is)"; nvidia-smi --query-gpu=memory.used --format=csv,noheader > $out/vram.txt
awk '/MemAvailable/{print}' /proc/meminfo > $out/memory-ready.txt
timeout 7200 python3 bench/sweep.py --api vllm --template deepseek --url http://127.0.0.1:30141 --card rtx3090 --config "D107-ec-kv1g-margin500" --prefill 8192 32768 --conc 1 2 4 --reps 3 --dec-reps 2 --vocab-max 128000 --no-early-exit --out $out/sweep.json > $out/sweep.log 2>&1; echo "sweep rc $?"
awk '/MemAvailable/{print}' /proc/meminfo > $out/memory-after.txt
grep -a "dsv41 cpu_tier" $out/server.log | tail -5
