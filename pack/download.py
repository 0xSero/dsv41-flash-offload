#!/usr/bin/env python3
"""Download the two pinned model inputs into MODEL_ROOT (directory names matter: the overlay pack symlinks into them).

  MODEL_ROOT/Mia-DeepSeek-V4.1-Flash-EXL3-3.0bpw   Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw @ c5534b90 (all files, 219.3 GB)
  MODEL_ROOT/DeepSeek-V4.1-Flash-engram            deepseek-ai/DeepSeek-V4.1-Flash @ 2cba9e42, only the Engram
                                                   n-gram tables (shards 47-48, 203.1 GB) + config + index
A token is read from HF_TOKEN by huggingface_hub if the repos ever become gated; it is never printed.
"""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

EXL3 = ("Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw", "c5534b90602b4090980a1c3ded1eb3b4d99a38d5",
        "Mia-DeepSeek-V4.1-Flash-EXL3-3.0bpw", None)
ENGRAM = ("deepseek-ai/DeepSeek-V4.1-Flash", "2cba9e42aa026125f3ed06c6d98c1db82f7ca027", "DeepSeek-V4.1-Flash-engram",
          ["config.json", "model.safetensors.index.json", "inference/engram.py",
           "model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"])

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, default=Path("/models"))
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
for repo, rev, name, allow in (EXL3, ENGRAM):
    out = a.root / name
    print(f"download {repo} @ {rev[:8]} -> {out}", flush=True)
    snapshot_download(repo_id=repo, revision=rev, local_dir=out, allow_patterns=allow, max_workers=a.workers)
print("download complete", flush=True)
