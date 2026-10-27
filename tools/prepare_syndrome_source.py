"""Small source extension; original matched quartet and all old files stay intact."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from proposals.paired_fault_modes import physical_counterparts
from tools.train_causal_contrast import load_pool
from causal_tsad.io import atomic_json,sha256,array_sha
SOURCE_SHA='1efe50fdb5f32fc0501bca05337851a3c92e6ff8bd6fce4969153513185b428b'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);data=load_pool(a.source,SOURCE_SHA);params={str(v['seed']):v for v in json.loads((a.source/'audit_parameters_NOT_MODEL_INPUT.json').read_text()).values()};files=[];generation_slots=0
 for split,limit in [('train',8),('calibration',4),('source_validation',8)]:
  for c in [8,12,16,20]:
   rows=sorted([r for r in data[split] if r['cases'].shape[-1]==c],key=lambda r:int(r['key']))[:limit]
   for row in rows:
    par=params[row['key']];assert par['root']==row['root'];perm=np.asarray(par['new_to_old']);g=(np.asarray(par['coefficient_before_permutation'])[perm][:,perm]!=0).astype(np.float32);assert np.array_equal(g,row['graph']);cases=physical_counterparts(row,par);assert np.array_equal(cases[:4],row['cases']);labels=np.zeros((8,1024),np.int8);labels[:4]=row['labels'];labels[4:,:256]=row['labels'][0,:256];labels[4:,256:]=1;p=a.output/(row['key']+'.npz');np.savez_compressed(p,cases=cases,graph=row['graph'],labels=labels)
    model_views=2 if split=='calibration' else 8;generation_slots+=cases.size;files.append(dict(path=p.name,SHA256=sha256(p),key=row['key'],split=split,C=c,family=row['family'],root=row['root'],prefix_contaminated=row['prefix_contaminated'],case_kind=['normal','normal','matched_measurement','matched_measurement','paired_measurement','paired_measurement','physical','physical'],model_views=model_views,input_cases_SHA256=array_sha(cases),graph_SHA256=array_sha(g),prefix_SHA256=array_sha(cases[0,:256]),labels_detail=dict(effect=None,anomaly_channel=None,anomaly_time=None)))
 atomic_json(a.output/'manifest.json',dict(status='complete',systems=80,files=files,parent_source_SHA256=SOURCE_SHA,materialized_scalar_slots=generation_slots,original_matched_quartet_exact=True,paired_measurement_physical_root_observations_exact=True,physical_amplitude_multiplier=1.5,simulator_parameters_used_only_in_builder=True,benchmark_arrays_opened=False))
