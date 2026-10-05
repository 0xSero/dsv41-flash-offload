#!/bin/bash
# C11 = C10 + B70 expert tier (host store in /dev/shm/dsv41bt shared with dsv41/ct/bt/bt_worker.py on B70 c3:00.0). C10 = D141 + decode routing capture ring (DSV41_ROUTE_RING, touch runs/D140-prof/route_dump to save). D130 = D128 with the long-prefill cap only while requests wait + 8 DMA slots; D128 = D126 + no throttling while requests wait + long-prefill threshold 7168 + torch profiler endpoint (idle unless /start_profile) (D125 + decode share during long prefills): mid-size steps on the CPU-tier split path, safe elastic EC, SMT CPU tier (research/12)
set -uo pipefail
cd ~/freetoken-exl3
out=${DSV41_RUN_DIR:-runs/C11-serve-$(date +%m%d-%H%M%S)}; mkdir -p $out; name=dsv41-lab
IMAGE=ghcr.io/0xsero/dsv41-flash-offload@sha256:964fb0a1e0f94fb307b757b39067e731fb9fa086b8e5c09d97eaf5639cc52d99
trap 'timeout 30 docker stop -t 5 dsv41-bt >/dev/null 2>&1; timeout 30 docker stop -t 15 $name >/dev/null 2>&1; rm -rf /dev/shm/dsv41bt' EXIT
rm -rf /dev/shm/dsv41bt; mkdir -m 777 /dev/shm/dsv41bt
awk '/MemAvailable/{print}' /proc/meminfo > $out/memory-before.txt
timeout 60 docker rm -f $name >/dev/null 2>&1 || true
mkdir -p $HOME/.cache/dsv41-lab
timeout 120 docker run -d --name $name --gpus device=0 --shm-size 8g --memory 300g --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -p 0.0.0.0:30141:8000 \
  -e DSV41_API_KEY_FILE=/run/secrets/api_key -v $HOME/freetoken-exl3/dsv41/serve.key:/run/secrets/api_key:ro \
  -e DSV41_MODEL_ROOT=/models -e DSV41_PACK=/models/DSV41-EXL3-3090-D010 -e DSV41_MAX_MODEL_LEN=262144 -e DSV41_KV_CACHE_BYTES=1610612736 \
  -e DSV41_LONG_PREFILL_WHEN_WAITING=${LONG_PREFILL:-7168} -e EXL3_HOST_DMA_SLOTS=${DMA_SLOTS:-8} -e DSV41_CT_HYB_MAX=${HYB_MAX:-2048} -e DSV41_CLAMP_MAX_TOKENS=1 -e DSV41_ROUTE_RING=${ROUTE_RING:-0} -e DSV41_CT_BUILD=/root/.cache/dsv41_ct_c11 -e DSV41_BT=${BT:-1} -e DSV41_BT_DIR=/dsv41bt -e EXL3_HOST_SHM_DIR=/dsv41bt -v /dev/shm/dsv41bt:/dsv41bt -e DSV41_DECODE_SHARE=${DECODE_SHARE:-0.25} -e DSV41_EC=1 -e DSV41_EC_ELASTIC=1 -e DSV41_EC_MARGIN_MB=1500 -e DSV41_EC_PREFILL_TOKENS=512 -e DSV41_EC_REWARM_AFTER=4 -e DSV41_EC_ASYNC=1 -e DSV41_CT_MAXBSZ=512 -e EXL3_HOST_DMA_MIN_TOKENS=513 -e DSV41_CT_MAXN=384 -e DSV41_CT_B=${CT_B:-0.20} -e DSV41_CT_TOK=0.35 -e DSV41_CT_THREADS=${CT_THREADS:-22} -e DSV41_CT_CPUS=${CT_CPUS:-2-23} -v $HOME/freetoken-exl3/dsv41/ct/cand11/ct_vllm.py:/opt/dsv41/ct/ct_vllm.py:ro -v $HOME/freetoken-exl3/dsv41/ct/cand11/ft_tier_cu_v.cu:/opt/dsv41/ct/ft_tier_cu_v.cu:ro -v $HOME/freetoken-exl3/dsv41/ct/k2b:/opt/dsv41/kernels/cpu_avx2:ro -v $HOME/freetoken-exl3/dsv41/ct/cand11/exl3bt/exl3.py:/usr/local/lib/python3.12/dist-packages/vllm_exl3/exl3.py:ro -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e DSV41_CPU_TIER=1 -e EXL3_HOST_DMA=1 -e EXL3_DMA_GEMM=1 \
  -e VLLM_EXL3_FAT_THRESHOLD=32 -e HF_HUB_OFFLINE=1 \
  -v $HOME/models:/models:ro -v $HOME/freetoken-exl3/runs/D140-prof:/out -v $HOME/.cache/dsv41-lab:/root/.cache \
  --entrypoint /opt/dsv41/docker/entrypoint.sh $IMAGE \
  --host 0.0.0.0 --port 8000 --served-model-name deepseek-v4.1-flash --max-model-len 262144 --kv-cache-memory 1610612736 \
  --max-num-batched-tokens 8192 --max-num-seqs 4 --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --profiler-config '{"profiler":"torch","torch_profiler_dir":"/out/prof","torch_profiler_with_stack":false}' ${PC_FLAG:---enable-prefix-caching} >/dev/null || { echo "docker run failed"; exit 2; }
( timeout 86400 docker logs -f $name > $out/server.log 2>&1 & )
K=$(cat $HOME/freetoken-exl3/dsv41/serve.key)
for i in $(seq 1 720); do
  curl -sf -m 5 -H "Authorization: Bearer $K" http://127.0.0.1:30141/v1/models >/dev/null && break
  timeout 10 docker inspect -f '{{.State.Running}}' $name 2>/dev/null | grep -q true || { echo "server died"; exit 3; }
  sleep 5
done
curl -sf -m 5 -H "Authorization: Bearer $K" http://127.0.0.1:30141/v1/models >/dev/null || { echo "server not ready"; exit 4; }
echo "ready $(date -Is)"; nvidia-smi --query-gpu=memory.used --format=csv,noheader > $out/vram.txt
awk '/MemAvailable/{print}' /proc/meminfo > $out/memory-ready.txt
( bench/xpu_run.sh 0000:c3:00.0 BT-$(basename $out) bash dsv41/ct/bt/bt_worker_run.sh $out >> $out/bt_launch.log 2>&1 & )
while timeout 10 docker inspect -f "{{.State.Running}}" $name 2>/dev/null | grep -q true; do sleep 30; done
echo "server exited $(date -Is)"
