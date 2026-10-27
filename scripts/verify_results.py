"""Recompute the published system/seed aggregation from actual saved records."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'configs/stat_temporal_v1.json').read_text());by=defaultdict(lambda:defaultdict(list));perseed={}
for seed in [348,349,350]:
    p=ROOT/f'results/source_v1/runs/true__stat_temporal__s{seed}_metrics.json';d=json.loads(p.read_text());m=defaultdict(list)
    assert len(d['metrics'])==512
    for row in d['metrics']:
        for key in ['AP','PR_AUC','FPR','conditional_Top1','conditional_MRR','joint_detected_root_recall']:
            if row.get(key) is not None:by[key][row['key']].append(row[key]);m[key].append(row[key])
    perseed[seed]={k:sum(v)/len(v) for k,v in m.items()}
means={k:sum(sum(v)/len(v) for v in systems.values())/len(systems) for k,systems in by.items()}
saved=json.loads((ROOT/'results/source_v1/summary.json').read_text())['summary']['true::stat_temporal::final']
for key,value in means.items():assert abs(value-saved[key]['mean'])<1e-12,(key,value,saved[key])
for row in cfg['seeds']:
    p=ROOT/row['checkpoint'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['checkpoint_SHA256']
p=ROOT/cfg['basis_checkpoint'];assert hashlib.sha256(p.read_bytes()).hexdigest()==cfg['basis_SHA256']
print(json.dumps(dict(status='passed',independent_systems=32,optimization_seeds=3,metrics=means,per_seed=perseed,not_prediction_ensemble=True),indent=2))
