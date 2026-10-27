"""One preserved source case: portable forward vs the original GPU score array."""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.model import CausalTemporalStatistics,ROOT
from proposals.improvement3.source import generate
from causal_tsad.io import array_sha

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default=None);p.add_argument('--output',type=Path);a=p.parse_args()
    with np.load(ROOT/'tests/confirmation_000.npz') as f:x=f['values'][2];g=f['graph'];all_values=f['values']
    regenerated=generate(0,'confirmation')
    source_exact=array_sha(regenerated['values'])==array_sha(all_values)
    source_error=float(np.max(np.abs(regenerated['values']-all_values)))
    # Windows/NumPy vs the original Linux runtime need not share libm bits.
    # Reproduction of published scores uses the preserved input, never this copy.
    assert source_error<1e-12,source_error
    assert np.array_equal(regenerated['graph'],g)
    with np.load(ROOT/'tests/confirmation_000.npz') as f:assert np.array_equal(regenerated['labels'],f['labels'])
    model=CausalTemporalStatistics(device=a.device);out=model.predict(x,g)
    with np.load(ROOT/'tests/reference_scores_s348.npz') as f:
        se=float(np.max(np.abs(out['score']-f['confirmation_000__2'])));re=float(np.max(np.abs(out['root_evidence']-f['root__confirmation_000__2'])))
    assert se<1e-7 and re<1e-7,(se,re)
    # The selected head must not consume the old TimeRCD raw/concat cache keys.
    from proposals.improvement3.components import components
    z=components(model.basis,torch.as_tensor(x,device=model.device),torch.as_tensor(g,device=model.device),'none')
    with torch.no_grad():
        b=model.head(z);zz=dict(z,raw_logit=torch.randn(len(x)-32,device=model.device,dtype=torch.float64)*1e6,concat_logit=torch.randn(len(x)-32,device=model.device,dtype=torch.float64)*1e6);c=model.head(zz)
    assert torch.equal(b['score'],c['score']) and torch.equal(b['root'],c['root'])
    record=dict(status='passed',torch_version=torch.__version__,numpy_version=np.__version__,device=str(model.device),source_array_regeneration_exact=source_exact,source_array_regeneration_max_error=source_error,
       score_max_error=se,root_evidence_max_error=re,unused_TimeRCD_keys_invariance_exact=True,
       global_updates=0,new_source_systems=0,scope='single existing confirmation view, packaging equivalence only')
    if a.output:a.output.write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps(record))
