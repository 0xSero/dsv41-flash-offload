#!/bin/bash
# Bake the measured D117/D118 runtime into the image (run once at build time, as root, no GPU needed):
#   1. vLLM source adaptations (runtime/patch_runtime.py) and the three vllm_exl3 prefill DMA patches, in the order the
#      campaign applied them at container start; every file's before/after sha256 must equal runtime/expected-receipts.json
#   2. the CPU expert tier + VRAM mirror cache hook (ct/ct_vllm.py) appended to vllm_exl3/exl3.py, same text as D118 inner.sh
#   3. the Engram disk-tier host callback library (dsv41/engram_disk.cpp -> build/engram_disk.so)
#   4. an import path that exposes only the two Python packages (ct, dsv41), not the repo's other directories
# usage: scripts/install.sh [ROOT=/opt/dsv41]
set -euo pipefail
ROOT=${1:-/opt/dsv41}
cd "$ROOT"
mkdir -p receipts build pypath
ln -sfn ../ct pypath/ct
ln -sfn ../dsv41 pypath/dsv41

python3 runtime/patch_runtime.py      > receipts/runtime-patch-receipt.json
python3 runtime/patch_dma_stock.py    > receipts/dma-patch-receipt.json
python3 runtime/patch_dma_segments.py > receipts/dma-segments-receipt.json
python3 runtime/patch_dma_gemm.py     > receipts/dma-gemm-receipt.json
python3 runtime/patch_dma_prefetch.py > receipts/dma-prefetch-receipt.json

P=$(python3 -c "import vllm_exl3,os;print(os.path.dirname(vllm_exl3.__file__))")
grep -q '_dsct.install' "$P/exl3.py" || printf '\ntry:\n    import sys as _s, ct.ct_vllm as _dsct\n    _dsct.install(_s.modules[__name__])\nexcept Exception as _e:\n    print("dsv41 ct install failed", repr(_e), flush=True)\n' >> "$P/exl3.py"

g++ -O3 -fPIC -shared -fopenmp -I/usr/local/cuda/include dsv41/engram_disk.cpp -L/usr/local/cuda/lib64 \
    -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -o build/engram_disk.so

python3 - "$ROOT" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
want = json.loads((root / "runtime/expected-receipts.json").read_text())
got = json.loads((root / "receipts/runtime-patch-receipt.json").read_text())["runtime_adaptations"]
bad = []
for r in got:
    w = want["vllm"].get(r["file"])
    if w != {"before": r["before"], "after": r["after"]}:
        bad.append((r["file"], r["before"], r["after"], w))
if len(got) != len(want["vllm"]):
    bad.append(("vllm file count", len(got), len(want["vllm"])))
for step, name in zip(want["vllm_exl3/exl3.py"], ("dma-patch-receipt", "dma-segments-receipt", "dma-gemm-receipt")):
    r = json.loads((root / "receipts" / (name + ".json")).read_text())
    if (r["before"], r["after"]) != (step["before"], step["after"]):
        bad.append((step["step"], r["before"], r["after"], step))
if bad:
    print("RECEIPT MISMATCH (image would not run the measured source):", *bad, sep="\n  ")
    sys.exit(1)
print(f"install: {len(got)} vLLM adaptations + 3 vllm_exl3 DMA patches match the D117/D118 receipts")
EOF
echo "install: CPU tier hook appended to $P/exl3.py; engram_disk.so built; PYTHONPATH=$ROOT/runtime:$ROOT/pypath"
