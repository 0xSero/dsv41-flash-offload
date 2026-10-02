"""Expand only native V4.1 projections that bypass quantization methods."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from exllamav3.modules.quant.exl3 import LinearEXL3
import argparse
ap=argparse.ArgumentParser()
ap.add_argument('--pack',type=Path,default=Path('/models/DSV41-EXL3-3090-D010'))
ap.add_argument('--source',type=Path,default=None,help='default: <pack parent>/Mia-DeepSeek-V4.1-Flash-EXL3-3.0bpw')
ap.add_argument('--result',type=Path,default=None,help='default: <pack>/native-dense-receipt.json')
a=ap.parse_args()
pack=a.pack
src=a.source or pack.parent/'Mia-DeepSeek-V4.1-Flash-EXL3-3.0bpw'
index=json.loads((pack/'model.safetensors.index.json').read_text())
wm=index['weight_map'];cfg=json.loads((pack/'config.json').read_text())
q=cfg['quantization_config'];layers=q['non_routed_exl3']['layers']
prefixes=sorted(k.removesuffix('.trellis') for k in wm if k.endswith('.trellis') and ('.attn.compressor.wkv.' in k or '.attn.compressor.wgate.' in k or '.attn.indexer.wk.' in k))
weights={};receipt=[]
for prefix in prefixes:
    with safe_open(src/wm[prefix+'.trellis'],framework='pt',device='cpu') as f:
        ts={s:f.get_tensor(prefix+'.'+s).cuda() for s in ('trellis','suh','svh','mul1')}
    linear=LinearEXL3(None,ts['suh'].numel(),ts['svh'].numel(),**ts)
    w=linear.get_weight_tensor().T.contiguous().cpu()
    assert torch.isfinite(w).all(),prefix
    weights[prefix+'.weight']=w
    for suffix in ('trellis','suh','svh','mul1'):del wm[prefix+'.'+suffix]
    wm[prefix+'.weight']='native-dense.safetensors'
    native='model.'+prefix
    if '.compressor.' in native:native=native.rsplit('.',1)[0]+'.fused_wkv_wgate'
    layers.pop(native,None)
    receipt.append({'key':prefix+'.weight','shape':list(w.shape),'dtype':str(w.dtype)})
    del linear,ts;torch.cuda.empty_cache()
save_file(weights,pack/'native-dense.safetensors')
q['non_routed_exl3']['layers']=layers
cfg['quantization_config']=q
if 'text_config' in cfg:cfg['text_config']['quantization_config']=q
(pack/'config.json').write_text(json.dumps(cfg,indent=2))
(pack/'quantization_config.json').write_text(json.dumps(q,indent=2))
(pack/'model.safetensors.index.json').write_text(json.dumps(index,indent=2))
result={'expanded':receipt,'bytes':sum(w.numel()*w.element_size() for w in weights.values()),'remaining_dense_quant_modules':len(layers)}
(a.result or pack/'native-dense-receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='expanded'},indent=2),flush=True)
