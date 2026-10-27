"""Check initialization, optimization exposure and shuffle against saved runs."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.model import ROOT,parameter_sha
from proposals.improvement3.models import CandidateMethod
cfg=json.loads((ROOT/'configs/stat_temporal_v1.json').read_text())
rows=json.loads((ROOT/'provenance/source_manifest.json').read_text())['rows']
pairs=[r for r in rows if r['split']=='train' for _ in range(6)]
assert len(pairs)==384
assert sum(2*1024*r['C'] for r in pairs)==11010048
assert sum(1024*r['C'] for r in pairs)==5505024
initial_checks=[]
for w in cfg['seeds']:
    seed=w['seed'];model=CandidateMethod('stat_temporal',seed)
    saved=json.loads((ROOT/f'results/source_v1/runs/true__stat_temporal__s{seed}.json').read_text())
    initial_checks.append(dict(seed=seed,actual=parameter_sha(model),original=saved['initial_SHA256'],exact=parameter_sha(model)==saved['initial_SHA256']))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==1794
    assert hashlib.sha256(np.random.default_rng(seed).permutation(384).tobytes()).hexdigest()==w['schedule_SHA256']
    model.load_state_dict(torch.load(ROOT/w['checkpoint'],map_location='cpu',weights_only=False)['model'])
    assert parameter_sha(model)==w['final_parameter_SHA256']
record=dict(status='passed_with_initialization_disclosure',initial_parameter_checks=initial_checks,seeds=[348,349,350],steps_per_seed=384,observed_per_seed=11010048,normal_targets_per_seed=5505024,trainable_parameters=1794,optimizer_steps_executed=0)
if '--output' in sys.argv:Path(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(record))
