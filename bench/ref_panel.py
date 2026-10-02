#!/usr/bin/env python3
"""Freeze and compare full-vocabulary teacher-forced distributions through native vLLM."""
import argparse,hashlib,json,time,urllib.request
from pathlib import Path
import numpy as np

QUESTIONS=[
 'Compute 17 * 23. Reply with only the integer.',
 'Name the chemical symbol for gold. Reply with only the symbol.',
 "Translate 'good morning' into Polish. Reply with only the translation.",
 'In Python, give a one-line expression that squares each item in xs. Reply with only code.',
 'Which comes first alphabetically: cobalt or copper? Reply with just the word.',
 'A train travels150km in2.5hours. Give its speed in km/h as just a number.',
 'Name the capital of Japan. Reply with only the city name.',
 'Return valid JSON containing key ok with value true. No other text.',
 'Explain why binary search needs sorted data in three sentences.',
 'Explain the difference between a process and a thread in three sentences.',
 'Explain why salt dissolves in water in three sentences.',
 'Give a short Python function that returns the greatest common divisor of two positive integers. Explain its loop briefly.',
]

def post(url,path,payload):
 assert not any(k in payload for k in ('max_tokens','max_completion_tokens','max_new_tokens'))
 req=urllib.request.Request(url+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=3600) as r:return json.load(r)

def completion(url,ids,score=False):
 payload={'model':'deepseek-v4.1-flash','prompt':ids,'temperature':0,'return_token_ids':True,'add_special_tokens':False}
 if score:payload['prompt_logprobs']=-1
 t=time.perf_counter();r=post(url,'/v1/completions',payload)
 choice=r['choices'][0]
 assert choice['finish_reason']=='stop',(choice['finish_reason'],choice['text'])
 return choice,r['usage'],time.perf_counter()-t

def distribution(choice,prompt_len,answer_len,vocab):
 data=choice['prompt_logprobs'];out=np.empty((answer_len,vocab),dtype=np.float32)
 assert len(data)>=prompt_len+answer_len,(len(data),prompt_len,answer_len)
 for j,row in enumerate(data[prompt_len:prompt_len+answer_len]):
  assert row is not None and len(row)==vocab,(j,None if row is None else len(row),vocab)
  for key,value in row.items():out[j,int(key)]=value['logprob']
  mass=float(np.exp(out[j].astype(np.float64)).sum())
  assert abs(mass-1)<0.01,(j,mass)
 return out

def metrics(ref,cmp):
 assert ref.shape==cmp.shape,(ref.shape,cmp.shape)
 p=np.exp(ref.astype(np.float64));positive=p>0
 delta=np.zeros_like(p);np.subtract(ref,cmp,out=delta,where=positive)
 kl=(p*delta).sum(axis=1)
 return kl,np.argmax(ref,axis=1)==np.argmax(cmp,axis=1)

ap=argparse.ArgumentParser()
ap.add_argument('--url',default='http://127.0.0.1:30141')
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--ref',type=Path)
ap.add_argument('--vocab',type=int,default=129280)
a=ap.parse_args();a.out.mkdir(exist_ok=True)
meta=[];all_kl=[];all_match=[]
if a.ref:
 manifest=json.loads((a.ref/'panel.json').read_text());questions=manifest['prompts']
else:
 (a.out/'questions.json').write_text(json.dumps(QUESTIONS,indent=2))
 questions=[{'question':q} for q in QUESTIONS]
for i,item in enumerate(questions):
 if a.ref:
  prompt_ids,answer_ids=item['prompt_ids'],item['answer_ids']
 else:
  text='<｜begin▁of▁sentence｜><｜User｜>'+item['question']+'<｜Assistant｜></think>'
  prompt_ids=post(a.url,'/tokenize',{'model':'deepseek-v4.1-flash','prompt':text,'add_special_tokens':False})['tokens']
  assert prompt_ids[0]==0 and prompt_ids.count(0)==1,prompt_ids
  c,usage,seconds=completion(a.url,prompt_ids)
  answer_ids=list(c['token_ids']);assert answer_ids
  if answer_ids[-1]==1:answer_ids.pop()
  assert answer_ids,(c,usage)
  item={**item,'prompt_ids':prompt_ids,'answer_ids':answer_ids,'answer':c['text'],'generation_usage':usage,'generation_s':seconds}
 (a.out/'current-prompt.json').write_text(json.dumps({'i':i,**item,'prompt_ids':prompt_ids,'answer_ids':answer_ids},indent=2))
 c,usage,seconds=completion(a.url,prompt_ids+answer_ids,True)
 lp=distribution(c,len(prompt_ids),len(answer_ids),a.vocab)
 file=f'prompt-{i:02d}.npz';np.savez_compressed(a.out/file,logprobs=lp)
 record={**item,'file':file,'positions':len(answer_ids),'score_usage':usage,'score_s':seconds,'sha256':hashlib.sha256((a.out/file).read_bytes()).hexdigest()}
 if a.ref:
  ref=np.load(a.ref/item['file'])['logprobs'];kl,match=metrics(ref,lp)
  record.update({'mean_kl_nats':float(kl.mean()),'top1_agreement':float(match.mean()),'p95_kl_nats':float(np.percentile(kl,95))})
  all_kl.extend(kl.tolist());all_match.extend(match.tolist())
 else:
  # The metric must detect deliberately wrong full-vocabulary probabilities.
  kl,match=metrics(lp,np.roll(lp,1,axis=1))
  assert kl.mean()>0.1 and match.mean()<0.1,'negative control did not fail'
  record['negative_control']={'mean_kl_nats':float(kl.mean()),'top1_agreement':float(match.mean())}
 record['prompt_sha256']=hashlib.sha256(json.dumps(prompt_ids+answer_ids).encode()).hexdigest()
 meta.append(record)
 (a.out/'panel.json').write_text(json.dumps({'vocab':a.vocab,'reference_path':'adapted native GPU EXL3/UVA + native FP8 Engram disk, BF16 compute, DSpark5','full_vocabulary':True,'prompts':meta},indent=2))
 print(f'[{i}] positions={len(answer_ids)} score_s={seconds:.2f}'+(f' KL={record["mean_kl_nats"]:.7g} top1={record["top1_agreement"]:.5f}' if a.ref else ' reference frozen'),flush=True)
if a.ref:
 rng=np.random.default_rng(196)
 counts=np.array([r['positions'] for r in meta]);sums=np.array([r['mean_kl_nats']*r['positions'] for r in meta])
 blocks=rng.integers(0,len(meta),size=(1000,len(meta)));ci=np.percentile(sums[blocks].sum(axis=1)/counts[blocks].sum(axis=1),[2.5,97.5]).tolist()
 result={'positions':len(all_kl),'prompts':len(meta),'full_vocabulary':True,'mean_kl_nats':float(np.mean(all_kl)),'median_kl_nats':float(np.median(all_kl)),'p95_kl_nats':float(np.percentile(all_kl,95)),'top1_agreement':float(np.mean(all_match)),'prompt_block_bootstrap95_nats':ci,'reference':str(a.ref)}
else:
 result={'positions':sum(r['positions'] for r in meta),'prompts':len(meta),'full_vocabulary':True,'negative_controls_passed':True,'reference_only':True}
(a.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
