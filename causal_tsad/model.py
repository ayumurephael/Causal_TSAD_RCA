"""Portable selected method. Only observed values and a normal DAG enter predict."""
from pathlib import Path
import json
import numpy as np
import torch
from causal_tsad.io import sha256
from proposals.causal_meta_basis import CausalMetaBasis
from proposals.improvement3.components import components
from proposals.improvement3.models import CandidateMethod

ROOT=Path(__file__).resolve().parents[1]

def parameter_sha(model):
    import hashlib
    h=hashlib.sha256()
    for name,v in sorted(model.state_dict().items()):h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def validate_input(values,graph):
    x=np.asarray(values,dtype=np.float64);g=np.asarray(graph,dtype=np.float64)
    assert x.ndim==2 and len(x)>=256 and np.isfinite(x).all(),'Need finite [T,C], T>=256'
    assert g.shape==(x.shape[1],)*2 and np.isin(g,[0,1]).all() and not g.diagonal().any()
    degree=g.sum(0).astype(int);q=list(np.flatnonzero(degree==0));count=0
    while q:
        n=q.pop();count+=1
        for child in np.flatnonzero(g[n]):
            degree[child]-=1
            if degree[child]==0:q.append(int(child))
    assert count==len(g),'Graph must be a normal DAG, source row -> target column'
    return x,g

class CausalTemporalStatistics:
    def __init__(self,seed=348,device=None,config=None):
        self.config=json.loads(Path(config or ROOT/'configs/stat_temporal_v1.json').read_text())
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        bp=ROOT/self.config['basis_checkpoint'];assert sha256(bp)==self.config['basis_SHA256']
        self.basis=CausalMetaBasis(348).to(self.device).eval().requires_grad_(False)
        self.basis.load_state_dict(torch.load(bp,map_location='cpu',weights_only=False)['model'])
        row=next(v for v in self.config['seeds'] if v['seed']==seed);p=ROOT/row['checkpoint'];assert sha256(p)==row['checkpoint_SHA256']
        self.head=CandidateMethod('stat_temporal',seed).to(self.device).eval().requires_grad_(False)
        self.head.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);self.threshold=row['threshold']
        self.before=(parameter_sha(self.basis),parameter_sha(self.head))

    @torch.inference_mode()
    def predict(self,values,graph):
        x,g=validate_input(values,graph)
        z=components(self.basis,torch.as_tensor(x,device=self.device),torch.as_tensor(g,device=self.device),'none')
        out=self.head(z)
        score=out['score'].cpu().numpy();nodes=out['root'].cpu().numpy()
        assert np.isfinite(score).all() and np.isfinite(nodes).all()
        assert (parameter_sha(self.basis),parameter_sha(self.head))==self.before
        return dict(score=score,root_evidence=nodes,root_probability=torch.softmax(out['root'],-1).cpu().numpy(),
                    first=32,alarm=score>self.threshold,threshold=self.threshold,context_audit=z['audit'])
