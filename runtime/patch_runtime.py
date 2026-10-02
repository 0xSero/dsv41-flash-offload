#!/usr/bin/env python3
"""Small, checked source adaptations for the pinned V4.1 Ampere runtime."""
import hashlib, json
from pathlib import Path
import vllm
base=Path(vllm.__file__).parent
receipt=[]
def edit(rel, changes):
    p=base/rel
    s=p.read_text()
    before=hashlib.sha256(s.encode()).hexdigest()
    for old,new in changes:
        assert s.count(old)==1,(rel,old,s.count(old))
        s=s.replace(old,new)
    compile(s,str(p),'exec')
    p.write_text(s)
    receipt.append({'file':rel,'before':before,'after':hashlib.sha256(s.encode()).hexdigest()})
edit('entrypoints/openai/completion/protocol.py',[
 ('    max_tokens: int | None = 16', '    max_tokens: int | None = None'),
])
edit('entrypoints/openai/completion/serving.py',[
 ('\n                    assert request.max_tokens is not None\n', '\n'),
 ('\n                assert request.max_tokens is not None\n', '\n'),
])
edit('model_executor/models/registry.py',[
 ('    "DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),',
  '    "DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),\n    "DeepseekV41LLMForCausalLM": ("vllm.models.deepseek_v4_1.nvidia.model", "DeepseekV41LLMForCausalLM"),'),
])
edit('models/deepseek_v4_1/nvidia/model.py',[
 ('            "layers.": "model.layers.",','            "head.": "lm_head.",\n            "layers.": "model.layers.",'),
 ('            "head.weight": "lm_head.weight",\n',''),
 ('                prefix=maybe_prefix(prefix, "lm_head"),','                quant_config=vllm_config.quant_config,\n                prefix=maybe_prefix(prefix, "lm_head"),'),
])
edit('models/deepseek_v4_1/common/engram.py',[
 ('        cpu_offload: bool = False,\n    ):','        cpu_offload: bool = False,\n        table_prefix: str | None = None,\n    ):'),
 ('class Engram(nn.Module):', 'import os\nif os.environ.get("DSV41_ENGRAM_DISK") == "1":\n    from dsv41.engram_disk import DiskEngramEmbedding as ParallelEngramEmbedding\n\nclass Engram(nn.Module):'),
 ('            cpu_offload=engram_config.cpu_offload if engram_config else True,', '            cpu_offload=engram_config.cpu_offload if engram_config else True,\n            table_prefix=f"{prefix}.embed",'),
 ('kwargs = {"device": "cpu", "pin_memory": True} if cpu_offload else {}','kwargs = {"device": "cpu", "pin_memory": False} if cpu_offload else {}'),
 ('        for param in (self.weight, self.weight_scale_inv):\n', '''        # Exact-size registration avoids the pinned allocator's power-of-two rounding.
        if cpu_offload:
            for param in (self.weight, self.weight_scale_inv):
                err = torch.cuda.cudart().cudaHostRegister(
                    param.data_ptr(), param.numel() * param.element_size(), 3)
                if "success" not in str(err).lower() and str(err) != "0":
                    raise RuntimeError(f"Engram cudaHostRegister failed: {err}")
                assert param.is_pinned(), "Engram registration was not recognized"
        for param in (self.weight, self.weight_scale_inv):
'''),
])
edit('model_executor/model_loader/default_loader.py',[
 ('import time\n', 'import time\nfrom pathlib import Path\n'),
 ('        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)', '''        # The overlay index omits original grouped wo_a slices and unused vision tensors.
        index_path = Path(hf_folder) / "model.safetensors.index.json"
        allowed = set(__import__("json").loads(index_path.read_text())["weight_map"]) if index_path.exists() else None
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator
                if allowed is None or name in allowed)'''),
])
print(json.dumps({'runtime_adaptations':receipt},indent=2),flush=True)
