#!/bin/bash
# dsv41-flash-offload container entrypoint.
#   docker run ... IMAGE [extra vllm serve args]   preflight, then serve the overlay pack (extra args are appended)
#   docker run ... IMAGE prepare                    download the pinned inputs (if missing) and build the overlay pack
#                                                   under /models (needs the GPU for ~10 min and a writable /models)
#   docker run ... IMAGE verify-pack                compare the overlay pack with the measured one (CPU only)
#   docker run ... IMAGE bash                       any other non-option command runs as-is
set -euo pipefail
ROOT=${DSV41_ROOT:-/opt/dsv41}
MODELS=${DSV41_MODEL_ROOT:-/models}
PACK=${DSV41_PACK:-$MODELS/DSV41-EXL3-3090-D010}
export DSV41_PACK=$PACK

log() { echo "[dsv41] $*" >&2; }
die() { echo "[dsv41] ERROR: $*" >&2; exit 1; }

gpu_check() {
    command -v nvidia-smi >/dev/null && timeout 20 nvidia-smi -L >&2 || die "no NVIDIA GPU visible (run with --gpus)"
    local ng vram
    ng=$(timeout 20 nvidia-smi -L | wc -l)
    [ "$ng" -eq 1 ] || log "WARNING: $ng GPUs visible; this build serves on ONE GPU (pass --gpus '\"device=N\"')"
    vram=$(timeout 20 nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    [ "${vram:-0}" -ge 23000 ] || die "GPU has ${vram} MiB; the measured configuration needs a 24 GB card"
}

case "${1:-}" in
    prepare)
        shift
        gpu_check
        [ -w "$MODELS" ] || die "$MODELS is not writable; mount the model root read-write for prepare"
        if [ ! -f "$MODELS/Mia-DeepSeek-V4.1-Flash-EXL3-3.0bpw/config.json" ] \
           || [ ! -f "$MODELS/DeepSeek-V4.1-Flash-engram/model-00048-of-00048.safetensors" ]; then
            FREE_GB=$(df -Pk "$MODELS" | awk 'NR==2{printf "%d", $4/1048576}')
            [ "$FREE_GB" -ge 400 ] || die "only ${FREE_GB} GiB free at $MODELS; inputs are 422 GB (393 GiB) + 3 GB pack"
            python3 "$ROOT/pack/download.py" --root "$MODELS"
        fi
        if [ -e "$PACK/config.json" ]; then
            log "pack already exists at $PACK; delete it to rebuild"
        else
            rm -rf "$PACK"   # a partial pack from an interrupted run
            python3 "$ROOT/pack/prepare_pack.py" --source "$MODELS/Mia-DeepSeek-V4.1-Flash-EXL3-3.0bpw" \
                --engram "$MODELS/DeepSeek-V4.1-Flash-engram" --out "$PACK"
            python3 "$ROOT/pack/expand_native_dense.py" --pack "$PACK"
        fi
        exec python3 "$ROOT/pack/verify.py" --pack "$PACK" ;;
    verify-pack)
        shift; exec python3 "$ROOT/pack/verify.py" --pack "$PACK" "$@" ;;
    -*|"") ;;
    *) exec "$@" ;;
esac

# ---- D130 runtime defaults; every value can be overridden with -e ------------------------------------------------
#   experts: all 15,360 routed experts pinned in host RAM (host-plan.json), read zero-copy / DMA-staged by the GPU;
#   prefill: 8,192-token chunks (16k chunks OOM on 24 GB); steps > 512 tokens stream every cold expert into 8 VRAM staging
#            slots x 8 experts (EXL3_HOST_DMA=1, next-layer prefetch), busy experts reconstructed to FP16 GEMMs
#            (EXL3_DMA_GEMM=1, fat threshold 32); steps <= 512 tokens (agent turns, admissions) run the decode split
#            path instead: CPU tier + zero-copy over the touched experts only (DSV41_CT_MAXBSZ=512, DMA_MIN_TOKENS=513);
#   decode:  VRAM mirror cache of the hottest experts (DSV41_EC=1, elastic: released before > 512-token steps, rewarmed
#            after 4 decode steps) + AVX2 CPU tier computing cold misses from the pinned copy (DSV41_CPU_TIER=1);
#   prefix caching on (V4.1's ratio-2 compressor ring is empty at block-aligned hits: exact, measured);
#   hybrid (D139): a lone 513..2048-token step streams most cold experts and lets the CPU tier compute the trailing
#            DMA batches in parallel (agent turns ~1.5-2 s faster);
#   scheduler: decode share 0.25 during long prefills, long-prompt chunks capped at 7168 only while >= 2 requests compete;
#   Engram n-gram tables: memory-mapped from NVMe, rows gathered per step by a CUDA host callback (DSV41_ENGRAM_DISK=1).
export DSV41_ENGRAM_DISK=${DSV41_ENGRAM_DISK:-1}
export DSV41_ENGRAM_LIB=${DSV41_ENGRAM_LIB:-$ROOT/build/engram_disk.so}
export EXL3_HOST_EXPERTS=${EXL3_HOST_EXPERTS:-$PACK/host-plan.json}
export VLLM_EXL3_MOE_KERNEL=${VLLM_EXL3_MOE_KERNEL:-exllamav3}
export VLLM_EXL3_FAT_THRESHOLD=${VLLM_EXL3_FAT_THRESHOLD:-32}
export EXL3_DMA_GEMM=${EXL3_DMA_GEMM:-1}
export EXL3_HOST_DMA=${EXL3_HOST_DMA:-1}
export EXL3_HOST_DMA_BATCH=${EXL3_HOST_DMA_BATCH:-8}
export EXL3_HOST_DMA_SLOTS=${EXL3_HOST_DMA_SLOTS:-8}
export EXL3_HOST_DMA_MIN_TOKENS=${EXL3_HOST_DMA_MIN_TOKENS:-513}
export EXL3_HOST_DMA_COLD_FUSED=${EXL3_HOST_DMA_COLD_FUSED:-1}
export DSV41_EC=${DSV41_EC:-1}
export DSV41_EC_MARGIN_MB=${DSV41_EC_MARGIN_MB:-1500}
export DSV41_EC_PREFILL_TOKENS=${DSV41_EC_PREFILL_TOKENS:-512}
export DSV41_EC_REWARM_AFTER=${DSV41_EC_REWARM_AFTER:-4}
export DSV41_EC_ASYNC=${DSV41_EC_ASYNC:-1}
export DSV41_EC_ELASTIC=${DSV41_EC_ELASTIC:-1}
export DSV41_CPU_TIER=${DSV41_CPU_TIER:-1}
export DSV41_CT_MAXBSZ=${DSV41_CT_MAXBSZ:-512}
export DSV41_CT_HYB_MAX=${DSV41_CT_HYB_MAX:-2048}   # lone 513..2048-token steps: CPU tier takes the trailing DMA batches
export DSV41_CLAMP_MAX_TOKENS=${DSV41_CLAMP_MAX_TOKENS:-1}   # near-full-window requests with max_tokens get the room left instead of HTTP 400
export DSV41_CT_MAXN=${DSV41_CT_MAXN:-384}
export DSV41_CT_B=${DSV41_CT_B:-0.20}
export DSV41_CT_TOK=${DSV41_CT_TOK:-0.35}
export DSV41_DECODE_SHARE=${DSV41_DECODE_SHARE:-0.25}
export DSV41_LONG_PREFILL_WHEN_WAITING=${DSV41_LONG_PREFILL_WHEN_WAITING:-7168}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export DSV41_CT_BUILD=${DSV41_CT_BUILD:-/root/.cache/dsv41_ct}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export VLLM_PLUGINS=${VLLM_PLUGINS:-vllm_exl3}
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
export FLASHINFER_DISABLE_VERSION_CHECK=1 TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-8.6}
export PYTHONUNBUFFERED=1

MAX_LEN=${DSV41_MAX_MODEL_LEN:-262144}
KV_BYTES=${DSV41_KV_CACHE_BYTES:-1610612736}     # 1.5 GiB = 297,224 fp8 tokens at 262144 (D119); 1 GiB is refused for 262144
CHUNK=${DSV41_MAX_NUM_BATCHED_TOKENS:-8192}
PORT=${PORT:-8000}

# API key: VLLM_API_KEY (env) or DSV41_API_KEY_FILE (a mounted file). vLLM reads VLLM_API_KEY itself, so the key never
# appears on a command line.
if [ -n "${DSV41_API_KEY_FILE:-}" ]; then
    [ -r "$DSV41_API_KEY_FILE" ] || die "DSV41_API_KEY_FILE=$DSV41_API_KEY_FILE is not readable"
    VLLM_API_KEY=$(tr -d '\r\n' < "$DSV41_API_KEY_FILE"); export VLLM_API_KEY
fi
[ -n "${VLLM_API_KEY:-}" ] || log "WARNING: no API key set (VLLM_API_KEY or DSV41_API_KEY_FILE); the endpoint is open"

# ---- preflight ----------------------------------------------------------------------------------------------------
gpu_check
ML=$(ulimit -l)
[ "$ML" = "unlimited" ] || log "WARNING: locked-memory limit is $ML KiB; run with --ulimit memlock=-1 (pinned host experts)"
if [ "$DSV41_CPU_TIER" = "1" ]; then
    for f in avx2 fma f16c; do
        grep -qw "$f" /proc/cpuinfo || die "CPU lacks $f; the CPU tier needs AVX2+FMA+F16C (or -e DSV41_CPU_TIER=0)"
    done
fi
NEED_GB=215   # measured: MemAvailable fell by 195-199 GiB from container start to ready (D030, D061, D107, D117)
AVAIL_GB=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
if [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != "max" ]; then
    LIM_GB=$(( $(cat /sys/fs/cgroup/memory.max) / 1073741824 ))
    [ "$LIM_GB" -ge "$NEED_GB" ] || die "container memory limit ${LIM_GB} GiB < ~${NEED_GB} GiB needed (raise --memory)"
fi
[ "$AVAIL_GB" -ge "$NEED_GB" ] || log "WARNING: MemAvailable ${AVAIL_GB} GiB < ~${NEED_GB} GiB this configuration pins; expect OOM"
for f in config.json host-plan.json model.safetensors.index.json; do
    [ -f "$PACK/$f" ] || die "no overlay pack at $PACK ($f missing); build it once with: docker run ... IMAGE prepare"
done
[ -e "$PACK/model-00048-of-00048.safetensors" ] || die "$PACK does not resolve the Engram shards; mount the whole model root at $MODELS"
log "host $(nproc) CPUs visible, MemAvailable ${AVAIL_GB} GiB, max-model-len $MAX_LEN, KV $KV_BYTES B, chunk $CHUNK"

ARGS=("$PACK" --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 1 --quantization exl3
      --served-model-name "${SERVED_NAME:-deepseek-v4.1-flash}" --max-logprobs -1
      --max-model-len "$MAX_LEN" --max-num-seqs 4 --max-num-batched-tokens "$CHUNK"
      --kv-cache-dtype fp8 --kv-cache-memory "$KV_BYTES" --gpu-memory-utilization 0.90
      --enable-prefix-caching --language-model-only --tokenizer-mode deepseek_v41 --trust-remote-code
      --disable-custom-all-reduce --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}'
      --cudagraph-capture-sizes 1 2 4 6 8 12 16 24
      --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41)
log "vllm serve ${ARGS[*]} $*"
exec vllm serve "${ARGS[@]}" "$@"
