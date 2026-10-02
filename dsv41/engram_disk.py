"""Read native FP8 Engram rows from mmap, with a CUDA-graph host callback.

Only selected rows are staged. The existing V4.1 dequantizer is reused unchanged.
"""
import ctypes, json, os, struct
from pathlib import Path
import numpy as np
import torch
from torch import nn
from vllm.models.deepseek_v4_1.common.engram import _engram_lookup_kernel
from vllm.model_executor.utils import set_weight_attrs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.config import get_current_vllm_config

_TABLES={}
class _Job(ctypes.Structure):
    _fields_=[(name,ctypes.c_void_p) for name in ('weight','scale','indices','out_weight','out_scale')]+[(name,ctypes.c_int64) for name in ('n','vocab_rows','dim','block')]
_LIB=ctypes.CDLL(os.environ.get('DSV41_ENGRAM_LIB','/opt/dsv41/build/engram_disk.so'))
_LIB.engram_enqueue.argtypes=[ctypes.c_size_t,ctypes.POINTER(_Job)]
_LIB.engram_enqueue.restype=ctypes.c_int

def _map_tensor(pack,key,index):
    path=pack/index[key]
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        meta=json.loads(f.read(n))[key]
    assert meta['dtype'] in ('F8_E4M3','F8_E4M3FN','F8_E8M0','F8_E8M0FNU','U8'),(key,meta)
    arr=np.memmap(path,mode='r',dtype=np.uint8,offset=8+n+meta['data_offsets'][0],shape=tuple(meta['shape']))
    return arr

@torch.library.custom_op('dsv41::engram_disk_lookup',mutates_args=('out',))
def _lookup(indices:torch.Tensor,out:torch.Tensor,table_id:int)->None:
    _TABLES[table_id]._run(indices,out)
@_lookup.register_fake
def _(indices,out,table_id):
    return None

class DiskEngramEmbedding(nn.Module):
    def __init__(self,num_embeddings,dim,head_sizes,block_size=32,cpu_offload=True,table_prefix=None):
        super().__init__()
        assert get_tensor_model_parallel_world_size()==1,'Disk tier currently supports the requested single GPU only'
        assert table_prefix is not None
        self.num_embeddings=num_embeddings
        self.dim=dim
        self.block_size=block_size
        self.n_hash_cols=len(head_sizes)
        self.part_n_hash_cols=self.n_hash_cols
        self.tp_size=1
        self.cpu_offload=True
        cfg=get_current_vllm_config()
        pack=Path(os.environ['DSV41_PACK'])
        index=json.loads((pack/'model.safetensors.index.json').read_text())['weight_map']
        raw=table_prefix.removeprefix('language_model.').removeprefix('model.')
        self._weights=_map_tensor(pack,raw+'.weight',index)
        self._scales=_map_tensor(pack,raw+'.scale',index)
        assert self._weights.shape==(num_embeddings,dim)
        assert self._scales.shape==(num_embeddings,dim//block_size)
        self.weight=nn.Parameter(torch.empty(0,dtype=torch.float8_e4m3fn,device='cpu'),requires_grad=False)
        self.weight_scale_inv=nn.Parameter(torch.empty(0,dtype=torch.uint8,device='cpu'),requires_grad=False)
        def loader(param,loaded):
            expected=self._weights.shape if param is self.weight else self._scales.shape
            assert tuple(loaded.shape)==expected,(loaded.shape,expected)
        for param in (self.weight,self.weight_scale_inv): set_weight_attrs(param,{'weight_loader':loader})
        self._capacity=cfg.scheduler_config.max_num_batched_tokens*self.n_hash_cols
        self._hi=torch.empty(self._capacity,dtype=torch.int64,device='cpu',pin_memory=True)
        self._hw=torch.empty(self._capacity,dim,dtype=torch.uint8,device='cpu',pin_memory=True)
        self._hs=torch.empty(self._capacity,dim//block_size,dtype=torch.uint8,device='cpu',pin_memory=True)
        self._gw=torch.empty_like(self._hw,device='cuda')
        self._gs=torch.empty_like(self._hs,device='cuda')
        self._ids=torch.arange(self._capacity,dtype=torch.int64,device='cuda')
        self._jobs={}
        self._table_id=len(_TABLES)
        _TABLES[self._table_id]=self
        print(f'Engram disk tier: {raw}, {num_embeddings} rows, staged capacity {self._capacity}',flush=True)

    def _run(self,indices,out):
        n=indices.numel()
        assert n<=self._capacity,(n,self._capacity)
        if n not in self._jobs:
            self._jobs[n]=_Job(self._weights.ctypes.data,self._scales.ctypes.data,self._hi.data_ptr(),self._hw.data_ptr(),self._hs.data_ptr(),n,self.num_embeddings,self.dim,self.block_size)
        # D2H -> CPU callback -> H2D -> unchanged GPU dequantizer, all on one stream.
        self._hi[:n].copy_(indices.flatten(),non_blocking=True)
        rc=_LIB.engram_enqueue(torch.cuda.current_stream().cuda_stream,ctypes.byref(self._jobs[n]))
        assert rc==0,f'cudaLaunchHostFunc: {rc}'
        self._gw[:n].copy_(self._hw[:n],non_blocking=True)
        self._gs[:n].copy_(self._hs[:n],non_blocking=True)
        ids=self._ids[:n].view_as(indices)
        grid=min((n+15)//16,torch.cuda.get_device_properties(0).multi_processor_count)
        _engram_lookup_kernel[(grid,)](self._gw,self._gs,ids,out,0,n,n,ids.stride(0),ids.stride(1),HEAD_START=0,LOCAL_HEADS=self.n_hash_cols,TOTAL_HEADS=self.n_hash_cols,DIM=self.dim,QUANT_BLOCK=self.block_size,BLOCK_R=16,GRID=grid)

    def lookup(self,indices,out,background=False):
        _lookup(indices,out,self._table_id)
    def forward(self,indices):
        out=torch.empty((*indices.shape,self.dim),dtype=torch.bfloat16,device=indices.device)
        self.lookup(indices,out)
        return out
