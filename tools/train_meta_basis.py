"""Finite source-only learning: equal plan, final checkpoint, no target arrays."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,argparse,traceback,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from proposals.causal_meta_basis import CausalMetaBasis,relation_loss
from causal_tsad.io import atomic_json,sha256,array_sha

ARMS=['frozen_basis','normal_only','normal_relation','wrong_graph_relation']
def nuisance(cases,key,name,step=None):
    if name=='plain':return cases.copy()
    rng=np.random.RandomState(int(key)%2**32 if step is None else 930048+step)
    scale=np.median(np.abs(cases[0,:256]-np.median(cases[0,:256],axis=0)),axis=0)*1.4826
    scale=np.where(scale<1e-3,1,scale)
    phase=rng.uniform(-np.pi,np.pi,cases.shape[2])
    if step is not None:amplitude=rng.uniform(0,4);period=rng.uniform(.5,1.25)
    else:amplitude=4.;period=.75 if name=='smooth' else 1.5
    b=amplitude*scale*np.sin(2*np.pi*np.arange(cases.shape[1])[:,None]/(cases.shape[1]*period)+phase)
    return cases+b[None]
def graph_for(g,key,mode):
    if mode=='true':return g.copy()
    if mode=='empty':return np.zeros_like(g)
    rng=np.random.RandomState((int(key)+743)%2**32);p=rng.permutation(len(g));gg=g[p][:,p]
    # The graph-only shuffle is fixed independently of values and labels.
    return gg.copy()
def psha(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def load(row,root):
    p=Path(root)/row['path'];assert sha256(p)==row['SHA256']
    with np.load(p) as f:cases=f['cases'].copy();g=f['graph'].copy()
    assert array_sha(cases)==row['input_cases_SHA256'] and array_sha(g)==row['graph_SHA256']
    return cases,g
def evaluate(m,rows,source,runroot,deadline):
    result=[];slots=factors=0;predseconds=0
    for row in rows:
        cases,g=load(row,source);views=2 if row['split']=='calibration' else 8
        for bg in (['plain','smooth'] if row['split']=='calibration' else ['plain','smooth','held_period']):
            values=nuisance(cases,row['key'],bg)[:views]
            assert all(sorted(array_sha(values[i,:,ch]) for i in [0,1])==sorted(array_sha(values[i,:,ch]) for i in [2,3]) for ch in range(row['C'])) if views==8 else True
            for mode in ['true','empty','rewired']:
                assert time.time()<deadline
                gg=graph_for(g,row['key'],mode);x=torch.tensor(values,dtype=torch.float64,device='cuda');graph=torch.tensor(gg,dtype=torch.float64,device='cuda')
                tick=time.perf_counter()
                with torch.no_grad():o=m(x,graph)
                torch.cuda.synchronize();predseconds+=time.perf_counter()-tick
                scores=o['score'].cpu().numpy();roots=o['root'].cpu().numpy();slots+=values.size;factors+=o['context_factorizations']
                metrics={};rca=[]
                if views==8:
                    for scope,ids in [('matched',[0,1,2,3]),('paired_measurement',[0,1,4,5]),('physical',[0,1,6,7])]:
                        y=np.repeat([0,0,1,1],704);metrics[scope]=dict(AP=float(average_precision_score(y,scores[ids].reshape(-1))),ROC_AUC=float(roc_auc_score(y,scores[ids].reshape(-1))))
                    for v in range(2,8):
                        order=np.argsort(-roots[v].mean(0),kind='stable');rank=int(np.flatnonzero(order==row['root'])[0])+1
                        rca.append(dict(kind=row['case_kind'][v],view=v,Top1=int(rank==1),Top3=int(rank<=min(3,row['C'])),MRR=1/rank))
                p=runroot/'evaluation'/row['split']/mode/(row['key']+'_'+bg);p.parent.mkdir(parents=True,exist_ok=True);assert not p.with_suffix('.npz').exists();np.savez_compressed(p.with_suffix('.npz'),scores=scores)
                record=dict(key=row['key'],split=row['split'],family=row['family'],C=row['C'],background=bg,mode=mode,metrics=metrics,RCA=rca,
                    input_SHA256=array_sha(values),graph_SHA256=array_sha(gg),source_file_SHA256=row['SHA256'],score_SHA256=sha256(p.with_suffix('.npz')),
                    normal_NLL=float(o['normal_nll'][:2].mean()),normal_whitened_square=float((o['residual'][:2].square()/o['variance'][:2]).mean()))
                atomic_json(p.with_suffix('.json'),record);result.append(dict(path=str(p.with_suffix('.json').relative_to(runroot)),SHA256=sha256(p.with_suffix('.json'))))
        atomic_json(runroot/'evaluation_progress.json',dict(completed=len(result),full_input_scalar_slots=slots,context_factorizations=factors))
    return result,dict(full_input_scalar_slots=slots,context_factorizations=factors,cuda_prediction_seconds=predseconds)
def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory)
    plan=json.loads(a.plan.read_text());assert sha256(Path(plan['source'])/'manifest.json')==plan['source_SHA256']
    manifest=json.loads((Path(plan['source'])/'manifest.json').read_text());rows={v['key']:v for v in manifest['files']};assert all(rows[k]['split']=='train' for k in plan['training_keys'])
    runroot=a.root/'runs'/(a.arm+'_s'+str(a.seed));runroot.mkdir(parents=True,exist_ok=False)
    m=CausalMetaBasis(a.seed).cuda();initial=psha(m);torch.save(dict(model=m.state_dict()),runroot/'initial.pth')
    if a.arm=='frozen_basis':
        for n,p in m.named_parameters():
            if n not in ['log_temperature','bias']:p.requires_grad_(False)
    opt=torch.optim.Adam(m.parameters(),lr=3e-4);curves=[];exposure=0;prediction_slots=0;start=time.time();plan_sha=sha256(a.plan)
    for step,key in enumerate(plan['training_keys']):
        assert time.time()<plan['deadline_unix'];row=rows[key];cases,g=load(row,plan['source']);v=nuisance(cases[:4],key,'plain' if step%2==0 else 'train_smooth',step=None if step%2==0 else step)
        assert all(sorted(array_sha(v[i,:,ch]) for i in [0,1])==sorted(array_sha(v[i,:,ch]) for i in [2,3]) for ch in range(row['C']))
        gg=graph_for(g,key,'rewired' if a.arm=='wrong_graph_relation' else 'true');m.train();opt.zero_grad(set_to_none=True)
        x=torch.tensor(v,dtype=torch.float64,device='cuda');graph=torch.tensor(gg,dtype=torch.float64,device='cuda')
        o=m(x,graph,detach_detection=a.arm in ['normal_only','frozen_basis']);nll=o['normal_nll'][:2].mean();relation=relation_loss(o['score'])
        loss=(nll if a.arm!='frozen_basis' else nll.detach())+.25*relation
        assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5));opt.step()
        exposure+=v.size;prediction_slots+=v.size;curves.append(dict(step=step+1,key=key,background='plain' if step%2==0 else 'train_smooth',input_SHA256=array_sha(v),graph_SHA256=array_sha(gg),loss=float(loss.detach()),normal_NLL=float(nll.detach()),relation_BCE=float(relation.detach()),gradient_norm=gn,observed_input_exposure=exposure))
        atomic_json(runroot/'progress.json',dict(status='training',updates=step+1,observed_input_exposure=exposure,plan_SHA256=plan_sha))
    assert len(curves)==128 and exposure==7340032
    m.eval().requires_grad_(False);final=psha(m);torch.save(dict(model=m.state_dict(),arm=a.arm,seed=a.seed,updates=128,input_exposure=exposure,initial_parameter_SHA256=initial,final_parameter_SHA256=final,plan_SHA256=plan_sha),runroot/'final.pth');atomic_json(runroot/'curves.json',curves)
    atomic_json(runroot/'progress.json',dict(status='source_evaluation',updates=128,observed_input_exposure=exposure))
    valrows=[v for v in manifest['files'] if v['split'] in ['calibration','source_validation']];assert len(valrows)==48
    records,cost=evaluate(m,valrows,plan['source'],runroot,plan['deadline_unix']);assert psha(m)==final
    atomic_json(runroot/'manifest.json',dict(status='complete',arm=a.arm,seed=a.seed,updates=128,observed_input_exposure=exposure,
        train_full_input_slots=prediction_slots,source_evaluation=cost,records=records,source_SHA256=plan['source_SHA256'],plan_SHA256=plan_sha,
        initial_parameter_SHA256=initial,final_parameter_SHA256=final,initial_checkpoint_SHA256=sha256(runroot/'initial.pth'),final_checkpoint_SHA256=sha256(runroot/'final.pth'),
        trainable_parameters=sum(p.numel() for p in CausalMetaBasis().parameters()) if a.arm!='frozen_basis' else 2,nominal_parameters=sum(p.numel() for p in m.parameters()),
        seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),benchmark_arrays_opened=False,target_updates=0,source_only=True,conditional_RCA=True,full_covariance_syndrome_implemented=False))
    atomic_json(runroot/'progress.json',dict(status='complete',updates=128,observed_input_exposure=exposure,records=len(records)))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','plan']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors'/(a.arm+'_s'+str(a.seed)+'.json'),dict(error=traceback.format_exc()));raise
