#!/usr/bin/env python3
"""Adapt a pinned native EXL3 pack for the V4.1 Ampere loader; never edit the source."""
import argparse, hashlib, json, re
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from exllamav3.modules.quant.exl3 import LinearEXL3

ap=argparse.ArgumentParser()
ap.add_argument("--source",type=Path,required=True)
ap.add_argument("--engram",type=Path,required=True)
ap.add_argument("--out",type=Path,required=True)
a=ap.parse_args()
a.out.mkdir(exist_ok=False)
idx=json.loads((a.source/"model.safetensors.index.json").read_text())
wm=idx["weight_map"]
metadata={}
for fn in sorted(set(wm.values())):
    with safe_open(a.source/fn,framework="pt",device="cpu") as f:
        for key in f.keys():
            metadata[key]=list(f.get_slice(key).get_shape())
for f in a.source.iterdir():
    if f.is_file() and f.name not in {"config.json","quantization_config.json","model.safetensors.index.json"}:
        (a.out/f.name).symlink_to(Path("..")/a.source.name/f.name)

layers={}
for key,shape in metadata.items():
    if not key.endswith(".trellis") or ".experts." in key or ".wo_a.slice." in key or key.startswith(("vision.","aligner.")):
        continue
    raw=key.removesuffix(".trellis")
    native=raw
    if raw=="head": native="lm_head"
    elif raw.startswith("layers."): native="model."+raw
    elif raw.startswith("mtp."):
        _,stage,rest=raw.split(".",2)
        native="model."+rest if rest.startswith("main_proj") else f"model.layers.{40+int(stage)}.{rest}"
    else: raise ValueError(f"Unmapped dense quant: {raw}")
    native=re.sub(r"\.shared_experts\.w[13]$",".shared_experts.gate_up_proj",native)
    native=re.sub(r"\.shared_experts\.w2$",".shared_experts.down_proj",native)
    native=re.sub(r"\.attn\.(wq_a|wkv)$",".attn.fused_wqa_wkv",native)
    native=re.sub(r"\.compressor\.(wkv|wgate)$",".compressor.fused_wkv_wgate",native)
    spec={"bits":shape[-1]//16}
    assert native not in layers or layers[native]==spec, (native,layers.get(native),spec)
    layers[native]=spec

q={"quant_method":"exl3","bits":3,"codebook":"mul1","scope":"dsv41_native_exl3","layer_bits":{str(i):4 for i in range(40,43)},"non_routed_exl3":{"codebook":"mul1","layers":layers},"non_routed_dtype_policy":"bf16_as_stored"}
cfg=json.loads((a.source/"config.json").read_text())
cfg["architectures"]=["DeepseekV41LLMForCausalLM"]
cfg["vision_config"]={}
cfg["quantization_config"]=q
if "text_config" in cfg: cfg["text_config"]["quantization_config"]=q
(a.out/"config.json").write_text(json.dumps(cfg,indent=2))
(a.out/"quantization_config.json").write_text(json.dumps(q,indent=2))
wm={k:v for k,v in wm.items() if ".attn.wo_a.slice." not in k and not k.startswith(("vision.","aligner.","image_"))}
reconstruction=[]
for root in [f"layers.{i}" for i in range(40)]+[f"mtp.{i}" for i in range(3)]:
    groups=[]
    for g in range(8):
        prefix=f"{root}.attn.wo_a.slice.{g}"
        fn=idx["weight_map"][prefix+".trellis"]
        with safe_open(a.source/fn,framework="pt",device="cpu") as f:
            ts={s:f.get_tensor(prefix+"."+s).cuda() for s in ("trellis","suh","svh","mul1")}
        linear=LinearEXL3(None,ts["suh"].numel(),ts["svh"].numel(),**ts)
        w=linear.get_weight_tensor().T.contiguous().cpu()
        assert torch.isfinite(w).all(),prefix
        groups.append(w)
        del linear,ts
    weight=torch.cat(groups,dim=0).contiguous()
    key=f"{root}.attn.wo_a.weight"
    fn="woa-"+root.replace(".", "-")+".safetensors"
    save_file({key:weight},a.out/fn)
    wm[key]=fn
    reconstruction.append({"key":key,"shape":list(weight.shape),"dtype":str(weight.dtype),"sha256":hashlib.sha256((a.out/fn).read_bytes()).hexdigest()})
    print(f"expanded {key}: {tuple(weight.shape)}",flush=True)
    del weight,groups
    torch.cuda.empty_cache()
base_idx=json.loads((a.engram/"model.safetensors.index.json").read_text())
engram={k:v for k,v in base_idx["weight_map"].items() if ".engram.embed." in k}
assert len(engram)==4,engram
for fn in set(engram.values()):
    (a.out/fn).symlink_to(Path("..")/a.engram.name/fn)
wm.update(engram)
(a.out/"model.safetensors.index.json").write_text(json.dumps({"metadata":{},"weight_map":wm},indent=2))
plan={f"model.layers.{i}.ffn.experts":list(range(384)) for i in range(40)}
(a.out/"host-plan.json").write_text(json.dumps(plan))
receipt={"source":str(a.source),"engram":str(a.engram),"out":str(a.out),"dense_quant_modules":len(layers),"weight_names":len(wm),"grouped_woa":reconstruction,"engram_keys":list(engram),"note":"Grouped wo_a reconstructed by pinned EXL3 GPU math to FP16; reference must use this same path."}
(a.out/"adapter-receipt.json").write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!="grouped_woa"},indent=2),flush=True)
