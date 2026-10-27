"""Input NPZ: values[T,C], graph[C,C]. No labels, roots or twins accepted."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from causal_tsad.model import CausalTemporalStatistics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,choices=[348,349,350],default=348);p.add_argument('--device',default=None);a=p.parse_args()
    assert not a.output.exists()
    with np.load(a.input) as f:
        assert set(f.files)=={'values','graph'},'Inference file must contain only observed values and normal DAG'
        x=f['values'].copy();g=f['graph'].copy()
    out=CausalTemporalStatistics(a.seed,a.device).predict(x,g)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**{k:v for k,v in out.items() if k!='context_audit'})
    print(dict(output=str(a.output),first=out['first'],points=len(out['score']),global_weight_updates=0))
