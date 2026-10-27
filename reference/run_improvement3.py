"""Bounded source campaign, all experimental modules implemented explicitly.

cache -> source-only finite training -> independent confirmation. Frozen
target transfer is a separate command requiring a source-only freeze record.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys,json,time,argparse,traceback,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from causal_tsad.io import atomic_json,sha256,array_sha
from causal_tsad.frozen_detector import FrozenDetector,parameter_sha
from proposals.causal_meta_basis import CausalMetaBasis
from proposals.context_crossfit import predict,configurations
from proposals.context_mixture import graph_input,ContextSelector,aggregate
from proposals.improvement3.source import generate
from proposals.improvement3.components import components
from proposals.improvement3.evidence import frozen,weights,mix,temporal_features
from proposals.improvement3.models import CandidateMethod,ARMS,loss
from proposals.improvement3.metrics import evaluate

WHITELIST=['mu','variance','response','features','valid','contribution','edges','raw_logit','concat_logit']

def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(),path
    tmp=path.with_suffix('.partial')
    with tmp.open('xb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)

def generate_job(pair):return generate(*pair)

def prepare_source(output):
    output.mkdir(parents=True,exist_ok=False);rows=[]
    jobs=[(i,split) for split,n in [('train',64),('calibration',16),('confirmation',32)] for i in range(n)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for r in pool.map(generate_job,jobs):
            row=r['row'];p=output/(row['key']+'.npz')
            save_npz(p,values=r['values'],graph=r['graph'],labels=r['labels'])
            row.update(path=p.name,SHA256=sha256(p),input_SHA256=array_sha(r['values']),graph_SHA256=array_sha(r['graph']),case_array_SHA256=[array_sha(v) for v in r['values']])
            atomic_json(output/(row['key']+'_audit.json'),r['audit']);rows.append(row)
    unique=sum(16*1024*r['C'] for r in rows if r['split']=='train')
    atomic_json(output/'manifest.json',dict(status='complete',rows=rows,train_unique_observed_scalars=unique,
      split_by_system=True,benchmark_arrays_read=False,source_namespace=348120000000))

def init(plan):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(plan['worker_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory)
    weight=Path(plan['weight']);assert sha256(weight)==plan['weight_SHA256']
    basis=CausalMetaBasis(348).cuda().eval().requires_grad_(False)
    basis.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False)['model'])
    detector=FrozenDetector(plan['detector_code'],plan['detector_weights'])
    return basis,detector

def cache_task(root,plan,task):
    start=time.time();basis,detector=init(plan);before=parameter_sha(basis)
    row=task['row'];src=Path(plan['source'])/row['path'];assert sha256(src)==row['SHA256']
    with np.load(src) as f:values=f['values'];labels=f['labels'];graph=f['graph']
    assert array_sha(values)==row['input_SHA256']
    g=graph_input(graph,row['key'],wrong=task['graph']=='wrong').cuda()
    if task['graph']=='empty':g.zero_()
    ident=task['id'];out=root/'cache'/ident;out.mkdir(parents=True,exist_ok=False)
    files=[];frozen_scores={};frozen_roots={};budgets=[];decomp_errors=[]
    old_selectors={}
    for item in plan.get('i11_frozen',[]):
        cp=Path(item['path']);assert sha256(cp)==item['SHA256']
        sel=ContextSelector(item['seed']).cuda().double().eval().requires_grad_(False)
        sel.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['selector']);old_selectors[item['name']]=sel
    for view,v in enumerate(values[:task.get('technical_smoke_views',len(values))]):
        if time.time()>plan['deadline_unix']:raise TimeoutError('finite campaign deadline')
        z=components(basis,torch.as_tensor(v,device='cuda',dtype=torch.float64),g,task['nuisance'])
        # Original raw input path is exactly unchanged; no causal tokens inside.
        donor=root/'cache'/(plan['primary_profile']+'__'+row['key'])/f'view{view}.npz'
        if task['profile']!=plan['primary_profile'] and donor.exists():
            with np.load(donor) as f:raw=f['raw_logit'].copy()
        else:raw=detector.predict(v,'selftrained_multi','native')['logit'][32:]
        z['raw_logit']=torch.as_tensor(raw,device='cuda',dtype=torch.float64)
        res=(z['response']-z['mu'][...,0])/z['variance'][...,0].sqrt()
        cat=np.concatenate([v[32:],res.cpu().numpy()],1)
        z['concat_logit']=torch.as_tensor(detector.predict(cat,'selftrained_multi','native')['logit'],device='cuda',dtype=torch.float64)
        target=(torch.as_tensor(values[row['normal_view'][view],32:],device='cuda')-z['center'])/z['scale']
        arr={k:z[k].cpu().numpy() for k in WHITELIST}
        p=out/f'view{view}.npz';save_npz(p,**arr,training_target=target.cpu().numpy(),labels=labels[view,32:])
        files.append(dict(view=view,path=str(p.relative_to(root)),SHA256=sha256(p),input_scalars=v.size))
        ss,rr,dd=frozen(z);decomp_errors.append(dd['decomposition_max_error'])
        if task['profile']==plan['primary_profile']:
            with torch.no_grad():
                for method_name,sel in old_selectors.items():
                    lw=sel(z['features'][...,:12],z['valid'])
                    ev=(((z['response'][...,None]-z['mu']).square()/(3*z['variance'])-.5*math.log(3))*lw.exp()).sum(-1)
                    ss[method_name]=aggregate(ev,float(basis.log_temperature.exp().clamp(.1,10)),float(basis.bias));rr[method_name]=ev
        native=((z['response']-z['mu'][...,0]).square()/(3*z['variance'][...,0])-.5*math.log(3))
        ss['full_native']=aggregate(native,float(basis.log_temperature.exp().clamp(.1,10)),float(basis.bias));rr['full_native']=native
        for k,s in ss.items():frozen_scores.setdefault(k,[]).append(s.cpu().numpy())
        for k,s in rr.items():frozen_roots.setdefault(k,[]).append(s.cpu().numpy())
        budgets.append(z['audit'])
        atomic_json(root/'task_progress'/(ident+'.json'),dict(stage='cache',view=view+1,total=16,seconds=time.time()-start))
    detector.verify_frozen();assert parameter_sha(basis)==before
    p=out/'frozen.npz';save_npz(p,**{k:np.stack(v) for k,v in frozen_scores.items()},**{'root__'+k:np.stack(v) for k,v in frozen_roots.items()})
    atomic_json(root/'records'/(ident+'.json'),dict(status='complete',task=task,files=files,frozen_path=str(p.relative_to(root)),frozen_SHA256=sha256(p),
      source_SHA256=sha256(src),basis_parameter_SHA256=before,basis_unchanged=True,graph_SHA256=array_sha(g.cpu().numpy()),
      graph_changed=not np.array_equal(g.cpu().numpy(),graph),decomposition_max_error=max(decomp_errors),
      budget=dict(causal_input_scalars=sum(b['input_scalars'] for b in budgets),factorizations=sum(b['factorizations'] for b in budgets),
                  feature_node_rows=sum(b['feature_node_rows'] for b in budgets),detector_input_scalars=detector.actual_input_scalars,
                  detector_padded_scalars=detector.padded_forward_scalars,detector_calls=detector.encoder_calls,training_exposure=0),
      seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),code=plan['code'],TimeRCD_RCA=False))

def cache_rows(root,plan,profile):
    rows=[]
    for task in plan['cache_tasks']:
        if task['profile']!=profile:continue
        rec=json.loads((root/'records'/(task['id']+'.json')).read_text());assert rec['status']=='complete'
        rows.append(rec)
    assert len(rows)==112
    return rows

def load(root,item):
    p=root/item['path'];assert sha256(p)==item['SHA256']
    with np.load(p) as f:
        z={k:torch.as_tensor(f[k],device='cuda') for k in WHITELIST}
        target=torch.as_tensor(f['training_target'],device='cuda');label=torch.as_tensor(f['labels'],device='cuda')
    return z,target,label

@torch.no_grad()
def evaluate_model(root,model,records):
    calibration=[];predictions=[];arrays={};exposure=0
    for rec in records:
        row=rec['task']['row']
        if row['split']=='train':continue
        for item in rec['files']:
            view=item['view']
            if row['split']=='calibration' and view%8>=2:continue
            z,_,label=load(root,item);p=model(z);s=p['score'].cpu().numpy();exposure+=item['input_scalars']
            assert np.isfinite(s).all()
            if row['split']=='calibration':calibration.append(s);continue
            nr=p['root'].cpu().numpy() if p['root'] is not None else None
            predictions.append((row,view,s,label.cpu().numpy(),nr))
            key=row['key']+'__'+str(view);arrays[key]=s
            if nr is not None:arrays['root__'+key]=nr
    threshold=float(np.quantile(np.concatenate(calibration),.99));metrics=[]
    for row,view,s,y,nr in predictions:
        metrics.append(dict(key=row['key'],C=row['C'],family=row['family'],view=view,background='plain' if view<8 else 'smooth',
          fault=row['fault_names'][view%8],**evaluate(s,y,threshold,nr,row['root'])))
    return dict(threshold=threshold,metrics=metrics,frozen_evaluation_exposure=exposure),arrays

def train_task(root,plan,task):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(plan['worker_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory)
    start=time.time();arm=task['arm'];seed=task['seed'];ident=task['id'];records=cache_rows(root,plan,task['profile'])
    model=CandidateMethod(arm,seed).cuda();initial=parameter_sha(model)
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.Adam(params,lr=.001)
    pairs=[]
    for rec in records:
        row=rec['task']['row']
        if row['split']!='train':continue
        for vi in range(2,8):
            # Exactly one plain/smooth condition per fault/system, matched for
            # ALL arms and graph controls; optimize seeds share the same data.
            bg=(int(row['key'].split('_')[-1])+vi)%2;v=bg*8+vi;n=row['normal_view'][v]
            pairs.append((rec['files'][n],rec['files'][v],row,vi))
    assert len(pairs)==384
    initial_eval,initial_arrays=evaluate_model(root,model,records)
    save_npz(root/'records'/(ident+'_initial_scores.npz'),**initial_arrays)
    atomic_json(root/'records'/(ident+'_initial.json'),initial_eval)
    rng=np.random.default_rng(seed);order=rng.permutation(len(pairs));curves=[];exposure=targets=0
    for step,idx in enumerate(order,1):
        if time.time()>plan['deadline_unix']:raise TimeoutError('fixed 384 steps or campaign deadline')
        ni,fi,row,vi=pairs[int(idx)];n,_,_=load(root,ni);f,target,label=load(root,fi)
        kind='relation' if vi in [2,3] else row['fault_names'][vi]
        optimizer.zero_grad();obj,diag=loss(model,n,f,target,label,row['root'],kind)
        assert torch.isfinite(obj);obj.backward()
        # Private root loss cannot rescale the detection gradient via a shared
        # global norm clip. Both task groups have their own matched norm cap.
        det_params=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('root_head.')]
        root_params=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith('root_head.')]
        norm=torch.nn.utils.clip_grad_norm_(det_params,1.)
        if root_params:torch.nn.utils.clip_grad_norm_(root_params,1.)
        optimizer.step()
        exposure+=ni['input_scalars']+fi['input_scalars']
        if arm.startswith(('context','query')):targets+=2*1024*row['C']
        elif arm.startswith('stat_'):targets+=1024*row['C']
        curves.append(dict(step=step,loss=float(obj.detach()),gradient_norm=float(norm),**diag,observed_exposure=exposure,key=row['key'],view=fi['view']))
        with (root/'task_progress'/(ident+'_updates.jsonl')).open('a') as f:
            f.write(json.dumps(curves[-1])+'\n');f.flush()
        if step%16==0:atomic_json(root/'task_progress'/(ident+'.json'),dict(stage='training',step=step,observed_exposure=exposure,seconds=time.time()-start))
    final,arrays=evaluate_model(root,model,records)
    cp=root/'records'/(ident+'_checkpoint.pth');assert not cp.exists()
    torch.save(dict(model=model.state_dict(),arm=arm,seed=seed,steps=384,observed_exposure=exposure,
                    normal_target_exposure=targets,profile=task['profile'],threshold=final['threshold'],code=plan['code']),cp)
    save_npz(root/'records'/(ident+'_scores.npz'),**arrays);atomic_json(root/'records'/(ident+'_metrics.json'),final)
    atomic_json(root/'records'/(ident+'_curves.json'),curves)
    atomic_json(root/'records'/(ident+'.json'),dict(status='complete',task=task,updates=384,observed_exposure=exposure,
      normal_target_exposure=targets,trainable_parameters=sum(p.numel() for p in params),initial_SHA256=initial,final_SHA256=parameter_sha(model),
      checkpoint_SHA256=sha256(cp),schedule_SHA256=__import__('hashlib').sha256(order.tobytes()).hexdigest(),
      evaluation_exposure=initial_eval['frozen_evaluation_exposure']+final['frozen_evaluation_exposure'],
      seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),basis_updated=False,TimeRCD_updated=False,
      benchmark_arrays_opened=False,TimeRCD_RCA=False,code=plan['code'],new_basis_forwards=0))

def frozen_evaluation(root,plan,profile):
    records=cache_rows(root,plan,profile);cal={};pred=[]
    for rec in records:
        row=rec['task']['row'];p=root/rec['frozen_path'];assert sha256(p)==rec['frozen_SHA256']
        if row['split']=='train':continue
        src=Path(plan['source'])/row['path']
        with np.load(src) as f:labels=f['labels'][:,32:]
        with np.load(p) as f:
            for method in f.files:
                if method.startswith('root__'):continue
                if row['split']=='calibration':cal.setdefault(method,[]).extend(f[method][[0,1,8,9]]);continue
                nodes=f['root__'+method] if 'root__'+method in f else [None]*16
                for view,s in enumerate(f[method]):pred.append((row,view,method,s.copy(),labels[view],None if nodes[view] is None else nodes[view].copy()))
    thresholds={k:float(np.quantile(np.concatenate(v),.99)) for k,v in cal.items()};out=[]
    for row,view,method,s,y,nr in pred:
        out.append(dict(key=row['key'],view=view,method=method,C=row['C'],family=row['family'],background='plain' if view<8 else 'smooth',
                        fault=row['fault_names'][view%8],**evaluate(s,y,thresholds[method],nr,row['root'])))
    atomic_json(root/'records'/('frozen_'+profile+'.json'),dict(status='complete',thresholds=thresholds,metrics=out,training_exposure=0))

def preflight(plan,output):
    # Exercise normal, missed and detected event records, including RCA.
    # r1 failed before optimization because event bounds were NumPy scalars.
    event_labels=np.array([0,0,1,1,0,0],dtype=np.int8)
    event_nodes=np.tile(np.array([[0.,1.]]),(6,1))
    event_probe=[evaluate(np.arange(6.),event_labels,t,event_nodes,1) for t in [2.,10.]]
    event_probe.append(evaluate(np.arange(6.),np.zeros(6,dtype=np.int8),2.))
    event_probe.append(evaluate(np.arange(6.),event_labels,2.))
    assert json.loads(json.dumps(event_probe,allow_nan=False))==event_probe
    assert event_probe[0]['events'][0]['onset']==2 and event_probe[0]['events'][0]['detected'] is True
    assert event_probe[1]['events'][0]['detected'] is False
    basis,det=init(plan);sample=generate(0,'confirmation');v=torch.as_tensor(sample['values'][0],device='cuda');g=torch.as_tensor(sample['graph'],device='cuda')
    z=components(basis,v,g);old=predict(basis,v,g,dict(configurations()['xfit'],first=32))
    residual_error=float(((z['response']-z['mu'][...,0])-old['residual']).abs().max())
    variance_error=float((z['variance'][...,0]-old['variance']).abs().max())
    # Same time/normalization/first32 equivalence to existing actual neural basis.
    assert residual_error<1e-8 and variance_error<1e-8
    raw=det.predict(v.cpu().numpy(),'selftrained_multi','native')['logit'][32:]
    z['raw_logit']=torch.as_tensor(raw,device='cuda',dtype=torch.float64);z['concat_logit']=z['raw_logit'].clone()
    ss,rr,dd=frozen(z);assert dd['decomposition_max_error']<1e-8
    z={k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in z.items()}
    errors={};grads={}
    for arm in ARMS:
        m=CandidateMethod(arm,348).cuda();o=m(z)
        if arm in ['protected','raw_adapt']:assert torch.equal(o['score'],z['raw_logit'])
        y=torch.zeros(len(v)-32,device='cuda');y[256:512]=1
        obj,_=loss(m,z,z,z['response'],y,sample['root'],'relation');obj.backward()
        grads[arm]=sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
        assert torch.isfinite(obj) and grads[arm]>0
    # Own-current intervention cannot enter same-time normal mean/query routing.
    vv=v.clone();vv[450,0]+=17
    zz=components(basis,vv,g)
    own_current_error=float((z['mu'][418,0]-zz['mu'][418,0]).abs().max())
    query_feature_error=float((z['features'][418,0]-zz['features'][418,0]).abs().max())
    assert own_current_error<1e-10 and query_feature_error<1e-10
    for nuisance in ['fourier','orthogonal']:
        zz=components(basis,v,g,nuisance);assert torch.isfinite(zz['mu']).all()
    # Equal private-head detector gradients, including context selector.
    aa=CandidateMethod('stat_temporal',348).cuda();bb=CandidateMethod('stat_typed',348).cuda()
    ga,_=loss(aa,z,z,z['response'],y,sample['root'],'relation');gb,_=loss(bb,z,z,z['response'],y,sample['root'],'relation');ga.backward();gb.backward()
    gradient_error=0.
    for key,param in aa.named_parameters():
        if param.grad is not None:gradient_error=max(gradient_error,float((param.grad-dict(bb.named_parameters())[key].grad).abs().max()))
    assert gradient_error<1e-10
    # End-to-end writer smoke includes an old I11 selector, the code path that
    # previously shadowed the task ID. This is technical work, not a train run.
    smoke_root=Path(output).parent/('smoke_'+Path(plan['code']).name)
    for d in ['records','cache','task_progress']:(smoke_root/d).mkdir(parents=True,exist_ok=True)
    campaign=Path(plan['code']).parent
    manifest=campaign/'source_pool/manifest.json'
    if manifest.exists():
        row=json.loads(manifest.read_text())['rows'][0]
        cp=next((campaign.parent/'i11_objective_alignment_v1/records').glob('*_relation_rank_s348_checkpoint.pth'))
        smoke_plan=dict(plan,source=str(manifest.parent),primary_profile='smoke',deadline_unix=time.time()+120,
            i11_frozen=[dict(name='i11_smoke_control',path=str(cp),SHA256=sha256(cp),seed=348)])
        task=dict(id='smoke_writer',profile='smoke',graph='true',nuisance='none',row=row,technical_smoke_views=1)
        cache_task(smoke_root,smoke_plan,task)
        writer=json.loads((smoke_root/'records/smoke_writer.json').read_text());assert writer['task']['id']=='smoke_writer' and len(writer['files'])==1
        writer_audit=dict(status='passed',record_SHA256=sha256(smoke_root/'records/smoke_writer.json'),budget=writer['budget'])
    else:writer_audit=dict(status='source_not_yet_prepared')
    det.verify_frozen()
    atomic_json(output,dict(status='passed',residual_equivalence_error=residual_error,variance_equivalence_error=variance_error,
      own_current_mean_error=own_current_error,own_current_query_feature_error=query_feature_error,
      decomposition=dd,gradient_L1=grads,zero_causal_increment_exact=True,
      technical_causal_input_scalars=5*v.numel(),technical_backward_calls=len(ARMS)+2,technical_detector_scalars=det.actual_input_scalars,
      private_RCA_detector_gradient_error=gradient_error,
      event_metrics_strict_JSON_roundtrip=True,
      end_to_end_writer=writer_audit,
      basis_weight_SHA256=plan['weight_SHA256'],optimizer_steps=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare_source','preflight','cache','train','frozen']);p.add_argument('--root',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path);p.add_argument('--task',type=int);p.add_argument('--profile');a=p.parse_args()
    if a.action=='prepare_source':prepare_source(a.output)
    else:
        plan=json.loads(a.plan.read_text())
        try:
            if a.action=='preflight':preflight(plan,a.output)
            elif a.action=='cache':cache_task(a.root,plan,plan['cache_tasks'][a.task])
            elif a.action=='train':train_task(a.root,plan,plan['train_tasks'][a.task])
            else:frozen_evaluation(a.root,plan,a.profile)
        except Exception:
            if a.root is not None:atomic_json(a.root/'errors'/(a.action+'_'+str(a.task)+'.json'),dict(error=traceback.format_exc()))
            raise
