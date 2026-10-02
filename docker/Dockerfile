# Local build of the dsv41-flash-offload image from this checkout:  docker build -f docker/Dockerfile -t dsv41-flash-offload .
# The published image (ghcr.io/0xsero/dsv41-flash-offload) is built by 0xSero/local-ai-images from the same steps,
# with this repository fetched at a pinned commit instead of COPY.
#
# Base + upstream pins = the campaign's runtime image dsv41-exl3:ampere-d005 (run D005), rebuilt as one Dockerfile:
#   lazymio/vllm-backport v0.13.0 (vLLM 0.13 with DeepSeek-V4.1, torch 2.13 cu130, sm80/86/89 build), amd64 manifest
#   Tokha233/deepseek-v4.1-flash-a100-turbo @ fae324ae  Ampere (sm80/sm86) file patches for vLLM's V4.1 path
#   turboderp-org/exllamav3 @ 5be88657                  EXL3 kernels, built here for sm_86
#   vcruz305/vllm-exl3 @ d3cfd394 + patches/vllm-exl3-host-dma.patch   EXL3 MoE plugin + host-expert / DMA hooks
FROM lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TORCH_CUDA_ARCH_LIST="8.6"
ARG MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN apt-get update \
 && apt-get install -y --no-install-recommends git patch libcusparse-dev-13-0 libcusolver-dev-13-0 \
 && rm -rf /var/lib/apt/lists/*

ARG A100_TURBO_COMMIT=fae324ae62ac5cef31b7d38f5d369618e1cae1fa
RUN git init -q /tmp/a100 && cd /tmp/a100 \
 && git fetch -q --depth 1 https://github.com/Tokha233/deepseek-v4.1-flash-a100-turbo "${A100_TURBO_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD && test "$(git rev-parse HEAD)" = "${A100_TURBO_COMMIT}" \
 && cp -a patches /opt/dsv41-ampere-patches && rm -rf /tmp/a100 \
 && python3 /opt/dsv41-ampere-patches/install.py

ARG EXLLAMAV3_COMMIT=5be886578ec80324c2c715269387be2058724b6e
RUN git clone -q https://github.com/turboderp-org/exllamav3.git /opt/exllamav3 \
 && git -C /opt/exllamav3 checkout -q "${EXLLAMAV3_COMMIT}" \
 && test "$(git -C /opt/exllamav3 rev-parse HEAD)" = "${EXLLAMAV3_COMMIT}" \
 && MAX_JOBS=${MAX_JOBS} python3 -m pip install --no-build-isolation --no-deps /opt/exllamav3 2>&1 | tail -3 \
 && python3 -m pip show exllamav3 >/dev/null

COPY . /opt/dsv41

ARG VLLM_EXL3_COMMIT=d3cfd394920360d69f820d2dc96f8292a9e10283
RUN git clone -q https://github.com/vcruz305/vllm-exl3.git /opt/vllm-exl3 \
 && git -C /opt/vllm-exl3 checkout -q "${VLLM_EXL3_COMMIT}" \
 && test "$(git -C /opt/vllm-exl3 rev-parse HEAD)" = "${VLLM_EXL3_COMMIT}" \
 && cd /opt/vllm-exl3 && patch -p1 < /opt/dsv41/patches/vllm-exl3-host-dma.patch \
 && VLLM_EXL3_NO_CUDA=1 python3 -m pip install --no-build-isolation --no-deps . \
 && python3 -m pip install --no-cache-dir marisa-trie==1.4.1

RUN chmod 0755 /opt/dsv41/docker/entrypoint.sh /opt/dsv41/scripts/install.sh \
 && /opt/dsv41/scripts/install.sh /opt/dsv41

ENV VLLM_PLUGINS=vllm_exl3 EXL3_MOE_KERNEL=exllamav3 \
    PYTHONPATH=/opt/dsv41/runtime:/opt/dsv41/pypath \
    DSV41_ENGRAM_LIB=/opt/dsv41/build/engram_disk.so
EXPOSE 8000
ENTRYPOINT ["/opt/dsv41/docker/entrypoint.sh"]
