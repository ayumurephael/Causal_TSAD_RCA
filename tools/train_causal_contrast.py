"""Fixed-budget cross-system TSAD/RCA contrast training on source quartets only."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,argparse,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from proposals.causal_contrast import CausalContrast
from proposals.matched_pairs import rewire
from proposals.raw_context import parameter_sha
from causal_tsad.io import atomic_json,sha256

def load_pool(path,expected):
 assert sha256(path/'manifest.json')==expected;m=json.loads((path/'manifest.json').read_text());assert m['status']=='complete' and m['full_univariate_class_multiset_exact_matching_groups']==448
 out={s:[] for s in ['train','calibration','source_validation']}
 for r in m['files']:
  assert sha256(path/r['path'])==r['SHA256']
  with np.load(path/r['path']) as f:
   for i in range(r['systems']):out[r['split']].append(dict(cases=f['cases'][i].copy(),labels=f['labels'][i].copy(),graph=f['graph'][i].copy(),root=int(f['root'][i]),key=str(int(f['seed'][i])),family=str(f['family'][i]),prefix_contaminated=bool(f['prefix_contaminated'][i])))
 assert [len(out[s]) for s in ['train','calibration','source_validation']]==[256,64,128]
 return out

def graph_for(row,mode):
 g=torch.from_numpy(row['graph']).cuda()
 return rewire(g,row['key']) if mode=='wrong' else g

def evaluate(model,data,mode,deadline=float('inf')):
 from sklearn.metrics import average_precision_score,roc_auc_score
 model.eval();normal=[];records={};queries=0
 with torch.no_grad():
  for r in data['calibration']:
   assert time.time()<deadline,'Campaign deadline during calibration'
   g=graph_for(r,mode)
   for k in [0,1]:
    v=torch.tensor(r['cases'][k],dtype=torch.float32,device='cuda');p=model(v,g,mode);normal.extend(p['logit'].cpu().tolist());queries+=704*v.shape[1]
  threshold=float(np.quantile(normal,.99))
  for r in data['source_validation']:
   assert time.time()<deadline,'Campaign deadline during validation'
   g=graph_for(r,mode);scores=[];normal_scores=[];roots=[];nll=[];outputs=[]
   for k in range(4):
    v=torch.tensor(r['cases'][k],dtype=torch.float32,device='cuda');p=model(v,g,mode);s=p['logit'].cpu().numpy();scores.append(s);outputs.append(s)
    if k<2:normal_scores.extend(s.tolist());nll.append(float(p['normal_NLL'].mean()))
    else:roots.append(int(p['root_logits'].mean(0).argmax())==r['root'])
    queries+=704*v.shape[1]
   y=np.repeat([0,0,1,1],704);score=np.concatenate(scores);null=float(np.max(np.abs(outputs[0]+outputs[1]-outputs[2]-outputs[3])))
   records[r['key']]=dict(channels=r['cases'].shape[-1],family=r['family'],prefix_contaminated=r['prefix_contaminated'],paired_AP=float(average_precision_score(y,score)),paired_ROC_AUC=float(roc_auc_score(y,score)),
    normal_FPR=float(np.mean(np.asarray(normal_scores)>threshold)),conditional_root_Top1=float(np.mean(roots)),normal_NLL=float(np.mean(nll)),quartet_logit_additivity_error=null,
    wrong_graph_changed_fraction=float((g.cpu().numpy()!=r['graph']).sum()/max(1,2*r['graph'].sum())))
 return dict(records=records,AP_macro=float(np.mean([r['paired_AP'] for r in records.values()])),source_threshold=threshold,extra_frozen_query_predictions=queries,
  scope='128 independent source mechanism quartets, not target benchmark; paired normal/fault query AP with50%positive prior; root Top1 is conditional on known full fault suffix')

def run(a):
 cfg=json.loads(a.config.read_text());torch.set_num_threads(1);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory)
 assert not a.output.exists();a.output.mkdir(parents=True);data=load_pool(a.source,a.source_sha256);model=CausalContrast().cuda();start=time.monotonic();initial=parameter_sha(model);counts={'linear':0}
 def hook(layer,inp,out):counts['linear']+=2*layer.in_features*out.numel()
 for layer in model.modules():
  if isinstance(layer,torch.nn.Linear):layer.register_forward_hook(hook)
 status=dict(status='initial_source_validation',mode=a.mode,seed=a.seed,pid=os.getpid(),local_commit=a.local_commit,github_commit=a.github_commit,source_manifest_SHA256=a.source_sha256,
  config_SHA256=sha256(a.config),parameters=sum(p.numel() for p in model.parameters()),initial_parameter_SHA256=initial,updates=0,
  optimized_observed_query_scalars=0,normal_anchor_query_scalars_included_not_extra=0,full_observed_input_scalar_exposures=0,extra_frozen_query_predictions=0,
  optimization_Linear_forward_FLOPs=0,extra_evaluation_Linear_forward_FLOPs=0,optimization_attention_forward_FLOPs=0,extra_evaluation_attention_forward_FLOPs=0,
  optimization_graph_aggregate_forward_FLOPs=0,extra_evaluation_graph_aggregate_forward_FLOPs=0,
  original_TimeRCD_loaded_or_updated=False,target_arrays_opened=False,metadata_or_normal_twin_as_model_input=False,
  RCA_gradient_scope='RCA head only; detached shared hidden, no interference with detector/predictor',training_root_supervision=True,
  compute_note='Dense Linear and attention matmul FLOPs; same perseed slots acrossgraph modes; backward/nonlinear/rootloss extra not equated')
 atomic_json(a.output/'progress.json',status)
 def ev(name):
  lin=counts['linear'];att=model.predictor.attention_forward_FLOPs;agg=model.graph_aggregate_forward_FLOPs;result=evaluate(model,data,a.mode,a.deadline);atomic_json(a.output/(name+'.json'),result)
  status['extra_frozen_query_predictions']+=result['extra_frozen_query_predictions'];status['extra_evaluation_Linear_forward_FLOPs']+=counts['linear']-lin;status['extra_evaluation_attention_forward_FLOPs']+=model.predictor.attention_forward_FLOPs-att
  status['extra_evaluation_graph_aggregate_forward_FLOPs']+=model.graph_aggregate_forward_FLOPs-agg
  return result
 ev('initial_source_validation');opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
 for epoch in range(3):
  for i in np.random.RandomState(cfg['data_seed']+epoch).permutation(256):
   assert time.monotonic()-start<3*3600 and time.time()<a.deadline
   row=data['train'][int(i)];g=graph_for(row,a.mode);model.train();opt.zero_grad(set_to_none=True);loss_sum=0.;lin=counts['linear'];att=model.predictor.attention_forward_FLOPs;agg=model.graph_aggregate_forward_FLOPs
   for view in range(4):
    v=torch.tensor(row['cases'][view],dtype=torch.float32,device='cuda');out=model(v,g,a.mode);label=torch.full_like(out['logit'],float(view>=2))
    loss=F.binary_cross_entropy_with_logits(out['logit'],label)
    if view<2:loss=loss+cfg['normal_NLL_weight']*out['normal_NLL'].mean();status['normal_anchor_query_scalars_included_not_extra']+=704*v.shape[1]
    else:loss=loss+cfg['root_loss_weight']*F.cross_entropy(out['root_logits'],torch.full((704,),row['root'],dtype=torch.long,device='cuda'))/np.log(v.shape[1])
    assert torch.isfinite(loss);(loss/4).backward();loss_sum+=float(loss)/4;status['optimized_observed_query_scalars']+=704*v.shape[1];status['full_observed_input_scalar_exposures']+=v.numel()
   grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad);opt.step();status['updates']+=1
   status['optimization_Linear_forward_FLOPs']+=counts['linear']-lin;status['optimization_attention_forward_FLOPs']+=model.predictor.attention_forward_FLOPs-att
   status['optimization_graph_aggregate_forward_FLOPs']+=model.graph_aggregate_forward_FLOPs-agg
   status.update(status='training',seconds=time.monotonic()-start,last_loss=loss_sum)
   if status['updates']==1 or status['updates']%16==0:
    atomic_json(a.output/'progress.json',status)
    with (a.output/'learning_curve.jsonl').open('a') as f:f.write(json.dumps({k:status[k] for k in ['updates','optimized_observed_query_scalars','last_loss','seconds']})+'\n')
  if epoch==0:ev('pass1_source_validation')
 assert status['updates']==768 and status['optimized_observed_query_scalars']==30277632 and status['full_observed_input_scalar_exposures']==44040192
 final=ev('final_source_validation');torch.save(dict(model=model.state_dict(),config=cfg,mode=a.mode,seed=a.seed,source_calibration=final,source_manifest_SHA256=a.source_sha256),a.output/'final.pth')
 status.update(status='complete',seconds=time.monotonic()-start,final_parameter_SHA256=parameter_sha(model),checkpoint_SHA256=sha256(a.output/'final.pth'),peak_cuda_bytes=torch.cuda.max_memory_allocated());atomic_json(a.output/'progress.json',status)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['source','config','output']:p.add_argument('--'+n,type=Path,required=True)
 for n in ['source-sha256','local-commit','github-commit']:p.add_argument('--'+n,required=True)
 p.add_argument('--mode',choices=['true','degree','wrong'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--deadline',type=float,default=float('inf'));a=p.parse_args()
 try:run(a)
 except Exception:
  a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
