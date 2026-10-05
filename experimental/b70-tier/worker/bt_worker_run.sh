#!/bin/bash
# B70 expert-tier worker for the cand11 server (runs under bench/xpu_run.sh 0000:c3:00.0 ...). RAM cap 40g, cores 24,25.
set -uo pipefail
cd ~/freetoken-exl3
node=$DSV41_XPU_RENDER
[ "$DSV41_XPU_PCI" = 0000:c3:00.0 ] && [ "$node" = /dev/dri/renderD129 ] || { echo "refuse: not c3"; exit 9; }
out=${1:?run dir}
timeout 10 docker rm -f dsv41-bt >/dev/null 2>&1 || true
exec docker run --rm --name dsv41-bt --network none --memory=40g --memory-swap=40g --ulimit memlock=-1:-1 --cpuset-cpus=24,25 \
  --device "$node:$node:rwm" -e ZE_AFFINITY_MASK=0 -e OMP_NUM_THREADS=1 -e BT_CPU=25 -e BT_MOE_LIB=/w/_moe-dsv41.so \
  -v $HOME/freetoken-exl3/dsv41/ct/bt:/w:ro -v /dev/shm/dsv41bt:/bt -v $HOME/models/Mia-DeepSeek-V4.1-Flash-EXL3-3.0bpw:/model:ro \
  --entrypoint python3 24c872759256 /w/bt_worker.py /bt /model ${BT_SLOTS:-2200} > $out/bt_worker.log 2>&1
