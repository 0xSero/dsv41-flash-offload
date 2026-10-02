#!/usr/bin/env python3
"""Check an overlay pack against the hashes of the pack the campaign measured on (pack/expected-sha256.json)."""
import argparse, hashlib, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--pack", type=Path, default=Path("/models/DSV41-EXL3-3090-D010"))
a = ap.parse_args()
want = json.loads((Path(__file__).resolve().parent / "expected-sha256.json").read_text())
bad = 0
for name, h in want["final_files"].items():
    got = hashlib.sha256((a.pack / name).read_bytes()).hexdigest()
    ok = got == h
    bad += not ok
    print(f"{'OK  ' if ok else 'DIFF'} {name} {got[:16]}")
receipt = json.loads((a.pack / "adapter-receipt.json").read_text())
for r in receipt["grouped_woa"]:
    ok = want["grouped_woa"].get(r["key"]) == r["sha256"]
    bad += not ok
    if not ok:
        print(f"DIFF {r['key']} {r['sha256'][:16]}")
print(f"{len(receipt['grouped_woa'])} grouped wo_a reconstructions checked")
missing = [k for k in ("../Mia-DeepSeek-V4.1-Flash-EXL3-3.0bpw", "../DeepSeek-V4.1-Flash-engram")
           if not (a.pack / k).is_dir()]
for m in missing:
    print(f"MISSING sibling directory {m} (the pack symlinks into it)")
sys.exit(1 if bad or missing else 0)
