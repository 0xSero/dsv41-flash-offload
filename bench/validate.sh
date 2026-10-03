#!/bin/bash
# Validate whatever config serves on :30141 -> runs/<tag>/. Gates: 0 soak errors, 0 OOM/EngineDead lines in the server log.
# usage: DSV41_KEY_FILE=~/dsv41.key bench/validate.sh <tag> <server.log> [soak minutes]
tag=${1:?tag}; srvlog=${2:?server log}; mins=${3:-30}
cd "$(dirname "$0")/.."; out=runs/$tag; mkdir -p $out; b=bench
python3 $b/ttft_curve.py --out $out/ttft.json > $out/ttft.log 2>&1
python3 $b/conc_bench.py --conc 1 2 4 --out $out/conc.json > $out/conc.log 2>&1
python3 $b/ctx_scan.py --lens 8192 65536 261000 --out $out/scan.json > $out/scan.log 2>&1
python3 $b/interleave.py --len 65536 --out $out/interleave.json > $out/interleave.log 2>&1
python3 $b/pc_test.py --lens 4096 32768 131072 --out $out/pc.json > $out/pc.log 2>&1
python3 $b/soak.py --workers 4 --minutes $mins --out $out/soak.json > $out/soak.log 2>&1; se=$?
oom=$(grep -ciE "OutOfMemory|EngineDead|EngineCore.*died" "$srvlog")
echo "{\"tag\":\"$tag\",\"soak_exit\":$se,\"oom_lines\":$oom,\"finished\":\"$(date -Is)\"}" | tee $out/gate.json
