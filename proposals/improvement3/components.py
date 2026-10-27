"""Actual frozen neural basis, protected context candidates and edge components.

Regression features contain own past and permitted current/past parents only.
Query reliability does NOT see the queried response, labels or generator data.
Time nuisance coefficients are fitted only outside the purged query block.
"""
import math
import torch
from proposals.context_crossfit import blocks, context_mask, design

@torch.inference_mode()
def components(model, values, graph, nuisance='none'):
    T, C = values.shape; dev = values.device; dt = values.dtype
    assert dt == torch.float64 and nuisance in ['none', 'fourier', 'orthogonal']
    t = torch.arange(32, T, device=dev); L = len(t); K = 5
    z0 = lambda *shape: torch.zeros(shape, device=dev, dtype=dt)
    mu = z0(L,C,K); var = torch.ones_like(mu); response = z0(L,C)
    center_out = z0(L,C); scale_out = torch.ones_like(center_out)
    valid = torch.zeros_like(mu, dtype=torch.bool); features = z0(L,C,K,20)
    edges = torch.nonzero(graph > 0).tolist(); E = len(edges)
    contribution = z0(L,E,K); factors = rows = 0
    bounds = torch.linspace(32,T,5).round().long().tolist()
    for node in range(C):
        pa = torch.where(graph[:,node]>0)[0].tolist()
        for lo, hi in blocks(T,8,'crossfit'):
            m = torch.as_tensor(context_mask(T,lo,hi,32,list(range(1,33)),'crossfit'), device=dev)
            qmask=(t>=lo)&(t<hi); ix=t[qmask]-32
            assert m.sum()>=64
            center=values[t[m]].median(0).values
            scale=(values[t[m]]-center).abs().median(0).values*1.4826
            scale=torch.where(scale<1e-3,torch.ones_like(scale),scale)
            z=(values-center)/scale; raw, ps=design(model,z,node,pa,t)
            # Four fixed low-frequency columns, no target-driven frequency search.
            if nuisance!='none':
                theta=2*math.pi*t.to(dt)/T
                tb=torch.stack([theta.sin(),theta.cos(),(2*theta).sin(),(2*theta).cos()],-1)
                if nuisance=='orthogonal':
                    # Remove projection onto legal causal design on CONTEXT only.
                    eye=torch.eye(raw.shape[-1],device=dev,dtype=dt)
                    coef=torch.linalg.solve(raw[m].T@raw[m]+eye,raw[m].T@tb[m])
                    tb=tb-raw@coef
                raw=torch.cat([raw,tb],-1)
            fm=raw[m].mean(0); fs=raw[m].std(0,unbiased=False).clamp_min(1e-3); fm[0]=0;fs[0]=1
            f=(raw-fm)/fs; q=f[qmask]; response[ix,node]=z[t[qmask],node]
            center_out[ix,node]=center[node];scale_out[ix,node]=scale[node]
            masks=[m]+[m&(t>=a)&(t<b) for a,b in zip(bounds[:-1],bounds[1:])]
            betas=[]; covs=[]
            for k,mask in enumerate(masks):
                if mask.sum()<64:betas.append(None);covs.append(None);continue
                cf=f[mask]; cy=z[t[mask],node];w=torch.ones_like(cy)
                eye=torch.eye(f.shape[-1],device=dev,dtype=dt)
                for _ in range(3):
                    chol=torch.linalg.cholesky(eye+cf.T@(w[:,None]*cf))
                    beta=torch.cholesky_solve((cf.T@(w*cy))[:,None],chol)[:,0]
                    err=cy-cf@beta
                    noise=((w*err.square()).sum()+beta.square().sum()+1)/(w.sum()+2)
                    noise=noise.clamp_min(1e-4);w=5/(4+err.square()/noise)
                covariance=noise*torch.cholesky_inverse(chol)
                vv=(q@covariance*q).sum(-1)
                mu[ix,node,k]=q@beta;var[ix,node,k]=noise+vv;valid[ix,node,k]=True
                parent_end=ps+13*len(pa)
                pv=cf[:,ps:parent_end]@beta[ps:parent_end] if pa else torch.zeros_like(cy)
                sl=lambda v:v.sign()*torch.log1p(v.abs())
                st=torch.stack([mask.sum().to(dt).log(),torch.as_tensor(f.shape[-1],device=dev,dtype=dt).log(),noise.log(),w.mean(),
                    torch.log1p(cf.square().mean()),torch.log1p(torch.cholesky_inverse(chol).trace()),
                    torch.log1p(beta[1:ps].norm()),sl(beta[0]),torch.as_tensor(len(pa),device=dev,dtype=dt),
                    torch.log1p(beta[ps:parent_end].norm()),torch.log1p(pv.var(unbiased=False)),sl(((pv-pv.mean())*(cy-cy.mean())).mean())])
                features[ix,node,k,:12]=st.clamp(-20,20)
                # Query-conditioned legal covariates, no current own response.
                features[ix,node,k,12]=torch.log1p(vv/noise)
                features[ix,node,k,13]=torch.log1p(q[:,1:ps].square().mean(-1))
                features[ix,node,k,14]=torch.log1p(q[:,ps:parent_end].square().mean(-1)) if pa else 0
                features[ix,node,k,15]=torch.log1p((q-cf.mean(0)).square().mean(-1))
                features[ix,node,k,16]=sl(q@beta)
                features[ix,node,k,17]=((q-cf.mean(0)).abs()>3*cf.std(0,unbiased=False).clamp_min(.01)).to(dt).mean(-1)
                for j,p in enumerate(pa):
                    e=edges.index([p,node]);start=ps+13*j
                    contribution[ix,e,k]=q[:,start:start+13]@beta[start:start+13]
                betas.append(beta);covs.append(covariance);factors+=3
            # Symmetric standardized coefficient-distance, not a classical Chow
            # F-test (ridge, overlap, autocorrelation invalidate that null).
            for k,b in enumerate(betas):
                if b is None:continue
                distances=[]
                for j,bj in enumerate(betas):
                    if j==k or bj is None or j==0 or k==0:continue
                    delta=b-bj;v=covs[k].diagonal()+covs[j].diagonal()
                    distances.append((delta.square()/v.clamp_min(1e-6)).mean())
                features[ix,node,k,18]=torch.stack(distances).median().log1p() if distances else 0
                features[ix,node,k,19]=torch.as_tensor(len(distances),device=dev,dtype=dt)
            rows+=len(t)
    assert valid[...,0].all() and torch.isfinite(mu).all() and torch.isfinite(var).all()
    return dict(mu=mu,variance=var,response=response,features=features,valid=valid,
                contribution=contribution,edges=torch.as_tensor(edges,device=dev,dtype=torch.int64).reshape(-1,2),
                center=center_out,scale=scale_out,
                audit=dict(factorizations=factors,feature_node_rows=rows,input_scalars=values.numel(),first=32))
