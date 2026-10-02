"""Owned sweep wrapper: retain terminal latency when the first result is EOS.

Empty EOS has no delivered token timestamp. Only prefill first-result timing
uses the terminal receipt; natural-EOS decode accounting stays untouched.
"""
import importlib.util,json,sys,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('campaign_sweep',Path(__file__).resolve().parent/'sweep.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
original=s.post
out=Path(sys.argv[sys.argv.index('--out')+1]).with_name('prefill-terminal-events.jsonl') if '--out' in sys.argv else None

def post(url,payload,first_only=False,timeout=7200):
 t0,times,last=original(url,payload,first_only,timeout)
 if first_only and not times:
  end=time.perf_counter();meta=(last or {}).get('meta_info',{})
  if meta.get('finish_reason',{}).get('type')!='stop':
   raise s.IncompleteStreamError('prefill returned no token and no natural EOS',{'last':last,'latency_s':end-t0})
  row={'timing_basis':'natural-EOS terminal receipt','delivered_token_ids':0,'completion_tokens':meta.get('completion_tokens'),'input_tokens':len(payload['input_ids']),'prompt_sha256':s.prompt_hash(payload['input_ids']),'terminal_latency_s':end-t0}
  if out is not None:
   with out.open('a') as f:f.write(json.dumps(row)+'\n')
  print('Prefill terminal receipt:',json.dumps(row),flush=True)
  times=[end]
 return t0,times,last
s.post=post
if __name__=='__main__':s.main()
