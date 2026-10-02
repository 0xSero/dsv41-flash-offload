#!/bin/bash
set -euo pipefail
cd ~/freetoken-exl3
out=runs/2026-10-01-D030-baseline-stability-p2
name=dsv41-d030-serve
mkdir -p dsv41/cache
trap 'timeout 30 docker stop -t 15 "$name" >/dev/null 2>&1 || true' EXIT
awk '/MemAvailable/{print}' /proc/meminfo > "$out/memory-before.txt"
timeout 120 docker run -d --name "$name" --gpus device=0 --network host --memory=300g --memory-swap=300g --cpus=24 --shm-size=8g --cap-add IPC_LOCK --ulimit memlock=-1:-1 -e PYTHONPATH=/w -e OMP_NUM_THREADS=16 -e DSV41_PACK=/models/DSV41-EXL3-3090-D010 -e DSV41_ENGRAM_DISK=1 -e EXL3_HOST_EXPERTS=/models/DSV41-EXL3-3090-D010/host-plan.json -e EXL3_MOE_KERNEL=exllamav3 -e VLLM_PLUGINS=vllm_exl3 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e TORCH_CUDA_ARCH_LIST=8.6 -e FLASHINFER_CUDA_ARCH_LIST=8.6 -e VLLM_EXL3_LOG_MOE_ROUTING=1 -v "$HOME/models:/models:ro" -v "$PWD:/w:ro" -v "$PWD/$out:/out" -v "$PWD/dsv41/cache:/root/.cache" --entrypoint bash dsv41-exl3:ampere-d005 /out/inner.sh > "$out/container-id.txt"

timeout 43200 docker logs -f "$name" > "$out/server.log" 2>&1 &
logs_pid=$!
for i in $(seq 1 480); do
    if curl -sf -m 5 http://127.0.0.1:30141/v1/models > "$out/models.json"; then break; fi
    timeout 10 docker inspect --format '{{.State.Running}}' "$name" | rg -q true || exit 3
    sleep 5
done
curl -sf -m 5 http://127.0.0.1:30141/v1/models > "$out/models.json"
python3 -u dsv41/resources.py --container "$name" --out "$out/resources.jsonl" > "$out/resources.log" 2>&1 &
mkdir -p "$out/repeat"
timeout 10800 docker exec "$name" python3 -u /w/dsv41/ref_panel.py --out /out/repeat --ref /w/runs/2026-10-01-D026-kv2-reference-p2/reference > "$out/repeat.log" 2>&1
python3 - "$out" <<'CHECK'
import json,sys,pathlib
out=pathlib.Path(sys.argv[1]);first=json.load(open('runs/2026-10-01-D026-kv2-reference-p2/repeat/result.json'));second=json.load(open(out/'repeat/result.json'))
assert second['prompts']==12 and second['positions']==416 and second['full_vocabulary']
quality={'role':'unchanged GPU baseline stability calibration before diagnostic P2','repeats':[first,second],'inherited_glm_guard':{'top1_min':0.988,'mean_kl_max':0.00103},'guard_results':[r['top1_agreement']>=0.988 and r['mean_kl_nats']<=0.00103 for r in (first,second)],'bit_exact':False,'optimization_accepted':False}
(out/'quality.json').write_text(json.dumps(quality,indent=2));print(json.dumps(quality,indent=2))
CHECK
timeout 21600 python3 -u bench/sweep.py --api vllm --template deepseek --url http://127.0.0.1:30141 --card dsv41_3090_exl3 --config D030-baseline-stability-KV2-dspark5 --vocab-max 128000 --prefill 8192 32768 --conc 1 2 4 --reps 3 --dec-reps 2 --no-early-exit --out "$out/sweep.json" > "$out/sweep.log" 2>&1
timeout 7200 python3 -u bench/stream_rate.py --api vllm --template deepseek --out "$out/stream.json" > "$out/stream.log" 2>&1
awk '/MemAvailable/{print}' /proc/meminfo > "$out/memory-after.txt"
printf 'COMPLETE diagnostic baseline stability/P2/stream; see quality.json\n' > "$out/result.txt"
