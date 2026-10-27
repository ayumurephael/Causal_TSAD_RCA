"""Portable source generation/cache/training/evaluation for stat_temporal only.

The mathematical modules/loss and 384-pair plan are copied byte-for-byte.
TimeRCD raw/concat forwards were irrelevant to this arm and are omitted.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from causal_tsad.io import atomic_json,sha256,array_sha

KEYS=['mu','variance','response','features','valid','contribution','edges']
def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
    with path.open('xb') as f:np.savez_compressed(f,**arrays)

def generate(a):
    from proposals.improvement3.source import generate as make
    a.source.mkdir(parents=True,exist_ok=False);rows=[]
    for split,n in [('train',64),('calibration',16),('confirmation',32)]:
        for i in range(n):
            r=make(i,split);row=r['row'];p=a.source/(row['key']+'.npz')
            save(p,values=r['values'],graph=r['graph'],labels=r['labels']);atomic_json(a.source/(row['key']+'_audit.json'),r['audit'])
            row.update(path=p.name,SHA256=sha256(p),input_SHA256=array_sha(r['values']),graph_SHA256=array_sha(r['graph']),case_array_SHA256=[array_sha(v) for v in r['values']]);rows.append(row)
    atomic_json(a.source/'manifest.json',dict(status='complete',rows=rows,train_unique_observed_scalars=sum(16*1024*r['C'] for r in rows if r['split']=='train'),split_by_system=True,benchmark_arrays_read=False,source_namespace=348120000000))

def cache(a):
    import torch
    from causal_tsad.model import ROOT,parameter_sha
    from proposals.causal_meta_basis import CausalMetaBasis
    from proposals.improvement3.components import components
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads((ROOT/'configs/stat_temporal_v1.json').read_text());p=ROOT/cfg['basis_checkpoint'];assert sha256(p)==cfg['basis_SHA256']
    model=CausalMetaBasis(348).to(a.device).eval().requires_grad_(False);model.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);before=parameter_sha(model)
    rows=json.loads((a.source/'manifest.json').read_text())['rows'];completed=[]
    for index,row in enumerate(rows):
        if index%a.world_size!=a.rank:continue
        src=a.source/row['path'];assert sha256(src)==row['SHA256'];record=a.cache/'records'/(row['key']+'.json')
        if record.exists():
            old=json.loads(record.read_text());assert old['source_SHA256']==row['SHA256']
            for f in old['files']:assert sha256(a.cache/f['path'])==f['SHA256']
            completed.append(row['key']);continue
        with np.load(src) as f:values=f['values'];labels=f['labels'];graph=f['graph']
        files=[];budgets=[]
        for view,x in enumerate(values):
            z=components(model,torch.as_tensor(x,device=a.device),torch.as_tensor(graph,device=a.device),'none')
            target=(torch.as_tensor(values[row['normal_view'][view],32:],device=a.device)-z['center'])/z['scale']
            dst=a.cache/'features'/row['key']/f'view{view}.npz'
            save(dst,**{k:z[k].cpu().numpy() for k in KEYS},training_target=target.cpu().numpy(),labels=labels[view,32:])
            files.append(dict(view=view,path=str(dst.relative_to(a.cache)),SHA256=sha256(dst),input_scalars=x.size));budgets.append(z['audit'])
        atomic_json(record,dict(status='complete',row=row,files=files,source_SHA256=row['SHA256'],basis_SHA256=cfg['basis_SHA256'],budget=budgets));completed.append(row['key'])
    assert parameter_sha(model)==before
    atomic_json(a.cache/f'cache_manifest_rank{a.rank}.json',dict(status='complete',keys=completed,world_size=a.world_size,rank=a.rank,source_manifest_SHA256=sha256(a.source/'manifest.json'),basis_parameter_SHA256=before,TimeRCD_forwards=0))

def load(cache,item,device):
    import torch
    p=cache/item['path'];assert sha256(p)==item['SHA256']
    with np.load(p) as f:return {k:torch.as_tensor(f[k],device=device) for k in KEYS},torch.as_tensor(f['training_target'],device=device),torch.as_tensor(f['labels'],device=device)

def evaluate(model,records,a,threshold=None):
    import torch
    from proposals.improvement3.metrics import evaluate as metrics
    cal=[];predictions=[];arrays={};exposure=0
    with torch.no_grad():
        for rec in records:
            row=rec['row']
            if row['split']=='train':continue
            for item in rec['files']:
                vi=item['view']
                if row['split']=='calibration' and vi%8>=2:continue
                z,_,y=load(a.cache,item,a.device);p=model(z);exposure+=item['input_scalars'];score=p['score'].cpu().numpy()
                if row['split']=='calibration':cal.append(score);continue
                node=p['root'].cpu().numpy();key=row['key']+'__'+str(vi);arrays[key]=score;arrays['root__'+key]=node
                predictions.append((row,vi,score,y.cpu().numpy(),node))
    fitted=float(np.quantile(np.concatenate(cal),.99));threshold=fitted if threshold is None else threshold
    rows=[dict(key=r['key'],C=r['C'],family=r['family'],view=v,background='plain' if v<8 else 'smooth',fault=r['fault_names'][v%8],**metrics(s,y,threshold,node,r['root'])) for r,v,s,y,node in predictions]
    return dict(threshold=threshold,recomputed_source_threshold=fitted,metrics=rows,frozen_evaluation_exposure=exposure),arrays

def learn_or_evaluate(a):
    import torch
    from causal_tsad.model import ROOT,parameter_sha
    from proposals.improvement3.models import CandidateMethod,loss
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads((ROOT/'configs/stat_temporal_v1.json').read_text());rows=json.loads((a.source/'manifest.json').read_text())['rows'];records=[]
    for r in rows:
        rec=json.loads((a.cache/'records'/(r['key']+'.json')).read_text());assert rec['source_SHA256']==r['SHA256'] and rec['basis_SHA256']==cfg['basis_SHA256'];records.append(rec)
    assert len(records)==112;a.output.mkdir(parents=True,exist_ok=False);m=CandidateMethod('stat_temporal',a.seed).to(a.device)
    start=time.monotonic();deadline=start+a.max_hours*3600
    if a.action=='evaluate':
        w=next(r for r in cfg['seeds'] if r['seed']==a.seed);p=ROOT/w['checkpoint'];assert sha256(p)==w['checkpoint_SHA256'];m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);m.eval().requires_grad_(False)
        result,arrays=evaluate(m,records,a,w['threshold']);save(a.output/'scores.npz',**arrays);atomic_json(a.output/'metrics.json',result);return
    initial=parameter_sha(m);ev,arr=evaluate(m,records,a);save(a.output/'initial_scores.npz',**arr);atomic_json(a.output/'initial_metrics.json',ev)
    pairs=[]
    for rec in records:
        row=rec['row']
        if row['split']!='train':continue
        for vi in range(2,8):
            bg=(int(row['key'].split('_')[-1])+vi)%2;v=bg*8+vi;n=row['normal_view'][v];pairs.append((rec['files'][n],rec['files'][v],row,vi))
    assert len(pairs)==384;order=np.random.default_rng(a.seed).permutation(384);opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001);curves=[];exposure=targets=0
    for step,idx in enumerate(order,1):
        assert time.monotonic()<deadline,'Fixed technical deadline; no automatic continuation'
        ni,fi,row,vi=pairs[int(idx)];n,_,_=load(a.cache,ni,a.device);f,target,label=load(a.cache,fi,a.device)
        opt.zero_grad();obj,diag=loss(m,n,f,target,label,row['root'],'relation' if vi in [2,3] else row['fault_names'][vi]);assert torch.isfinite(obj);obj.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for name,p in m.named_parameters() if p.requires_grad and not name.startswith('root_head.')],1);opt.step()
        exposure+=ni['input_scalars']+fi['input_scalars'];targets+=1024*row['C'];curves.append(dict(step=step,loss=float(obj.detach()),gradient_norm=float(norm),**diag,observed_exposure=exposure,key=row['key'],view=fi['view']))
    assert exposure==11010048 and targets==5505024
    result,arrays=evaluate(m,records,a);save(a.output/'scores.npz',**arrays);atomic_json(a.output/'metrics.json',result);atomic_json(a.output/'curves.json',curves)
    cp=a.output/'checkpoint.pth';torch.save(dict(model=m.state_dict(),arm='stat_temporal',seed=a.seed,steps=384,observed_exposure=exposure,normal_target_exposure=targets,threshold=result['threshold']),cp)
    atomic_json(a.output/'manifest.json',dict(status='complete',optimizer_steps=384,observed_exposure=exposure,normal_target_exposure=targets,checkpoint_SHA256=sha256(cp),schedule_SHA256=__import__('hashlib').sha256(order.tobytes()).hexdigest(),initial_parameter_SHA256=initial,final_parameter_SHA256=parameter_sha(m),source_manifest_SHA256=sha256(a.source/'manifest.json'),seconds=time.monotonic()-start,TimeRCD_forwards=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','cache','train','evaluate']);p.add_argument('--source',type=Path,required=True);p.add_argument('--cache',type=Path,default=Path('runs/cache'));p.add_argument('--output',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,choices=[348,349,350],default=348);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--max-hours',type=float,default=6);a=p.parse_args()
    if a.action=='generate':generate(a)
    elif a.action=='cache':cache(a)
    else:assert a.output is not None;learn_or_evaluate(a)
