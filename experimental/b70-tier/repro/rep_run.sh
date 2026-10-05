#!/bin/bash
# cand11 segfault repro without GPU0. usage: rep_run.sh <tag> [env...]   e.g. rep_run.sh huge1 REP_HUGE=1 REP_MIN=40
# CPU side: dsv41 image, NO GPUs, cpuset 26-47 (+ store build), 40g. B70 side: bt_worker on c3:00.0 via xpu_run.sh.
# A watcher pauses the repro container whenever GPU0 (the live server) is busy.
set -uo pipefail
cd ~/freetoken-exl3
tag=${1:?tag}; shift
out=runs/REP-$tag-$(date +%m%d-%H%M%S); mkdir -p $out
envs=(); for kv in "$@"; do envs+=(-e "$kv"); done
D=/dev/shm/dsv41rep; mkdir -p $D; chmod 777 $D
IMG=ghcr.io/0xsero/dsv41-flash-offload@sha256:964fb0a1e0f94fb307b757b39067e731fb9fa086b8e5c09d97eaf5639cc52d99
cleanup() { timeout 20 docker rm -f dsv41-rep dsv41-rep-bt >/dev/null 2>&1; kill $WPID 2>/dev/null; }
trap cleanup EXIT
timeout 20 docker rm -f dsv41-rep dsv41-rep-bt >/dev/null 2>&1
rm -f $D/meta.json $D/mbox.bin
timeout 10 docker run -d --name dsv41-rep --network none --memory=40g --memory-swap=40g --cpuset-cpus=26-47 \
  -v $HOME/freetoken-exl3/dsv41/ct/k2b:/k:ro -v $D:/bt -v $HOME/freetoken-exl3/runs/D140-prof/routes_1004-164629_1522920.npy:/routes.npy:ro \
  -v $HOME/models/Mia-DeepSeek-V4.1-Flash-EXL3-3.0bpw:/model:ro -v $HOME/freetoken-exl3/dsv41/ct/rep:/r:ro -v $HOME/.cache/dsv41-rep-build:/build \
  "${envs[@]}" --entrypoint python3 $IMG /r/rep_ct.py >/dev/null || { echo "rep container failed"; exit 2; }
( docker logs -f dsv41-rep > $out/rep.log 2>&1 ) &
# B70 worker once meta.json exists
( for i in $(seq 1 900); do [ -f $D/meta.json ] && break; sleep 2; done
  [ -f $D/meta.json ] && bench/xpu_run.sh 0000:c3:00.0 dsv41-rep bash -c "
    timeout 10 docker rm -f dsv41-rep-bt >/dev/null 2>&1
    exec docker run --rm --name dsv41-rep-bt --network none --memory=40g --memory-swap=40g --ulimit memlock=-1:-1 --cpuset-cpus=24,25 \
      --device \$DSV41_XPU_RENDER:\$DSV41_XPU_RENDER:rwm -e ZE_AFFINITY_MASK=0 -e OMP_NUM_THREADS=1 -e BT_CPU=25 -e BT_MOE_LIB=/w/_moe-dsv41.so \
      -v $HOME/freetoken-exl3/dsv41/ct/bt:/w:ro -v $D:/bt -v $HOME/models/Mia-DeepSeek-V4.1-Flash-EXL3-3.0bpw:/model:ro \
      --entrypoint python3 24c872759256 /w/bt_worker.py /bt /model \${BT_SLOTS:-300}" > $out/bt_worker.log 2>&1 ) &
# watcher: pause the repro while the live server's GPU is busy
( paused=0; while timeout 5 docker inspect dsv41-rep >/dev/null 2>&1; do
    u=$(timeout 5 nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ "${u:-0}" -gt 5 ] && [ $paused = 0 ]; then timeout 10 docker pause dsv41-rep >/dev/null 2>&1 && paused=1 && echo "$(date +%T) pause (gpu0 $u%)" >> $out/watch.log
    elif [ "${u:-0}" -le 5 ] && [ $paused = 1 ]; then sleep 20; timeout 10 docker unpause dsv41-rep >/dev/null 2>&1; paused=0; echo "$(date +%T) resume" >> $out/watch.log; fi
    sleep 3; done ) & WPID=$!
timeout 7200 docker wait dsv41-rep > $out/exit.txt 2>&1
sleep 2; echo "exit $(cat $out/exit.txt)"; tail -n 5 $out/rep.log
