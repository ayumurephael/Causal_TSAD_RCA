"""Frozen normal basis, purged offline context inference and separate online track.

Only X and the normal graph enter this module. No event, label, twin or generator
parameter is accepted. Cross-fitting is transductive latent inference, not an
online claim. The predictive-variance formula is matched to C04l.
"""
import math
import numpy as np
import torch
from scipy.signal import find_peaks

FIRST=320

def configurations():
    main=dict(mode='crossfit',K=8,gap=32,nu=4.,normalization='fold',periodic=False,long_parent=False)
    out={'xfit':main,'online':dict(main,mode='online')}
    for k in [4,16]:out['K'+str(k)]=dict(main,K=k)
    for gap in [0,16,64,128]:out['gap'+str(gap)]=dict(main,gap=gap)
    for nu in [2.,8.,None]:out['nu'+str(nu)]=dict(main,nu=nu)
    for norm in ['prefix','global']:out['norm_'+norm]=dict(main,normalization=norm)
    out['period']=dict(main,periodic=True)
    out['parent32']=dict(main,long_parent=True)
    out['period_parent32']=dict(main,periodic=True,long_parent=True)
    return out

def periods(x):
    """Offline ACF, no labels. Bounded 8..min(1500,T//4); no peak -> disabled.
    Missing long-lag entries are zero with a validity indicator, never future fill.
    """
    result=[]
    for v in x.T:
        v=v-np.median(v);n=len(v);size=1<<(2*n-1).bit_length()
        f=np.fft.rfft(v,n=size);a=np.fft.irfft(f*f.conj(),n=size)[:n]
        if a[0]<=1e-20:result.append(0);continue
        a=a/a[0];hi=min(1500,n//4);p,_=find_peaks(a[8:hi+1]);p=p+8
        result.append(int(p[np.argmax(a[p])]) if len(p) else 0)
    return result

def context_mask(length,lo,hi,gap,lag_support,mode):
    t=np.arange(32,length)
    if mode=='online':return t<lo-gap
    keep=(t<lo-gap)|(t>=hi+gap)
    # No held-out response may re-enter coefficient fitting through lag features.
    # Purge actual lags as well as the requested statistical gap; report counts.
    for lag in lag_support:keep&=~((t-lag>=lo)&(t-lag<hi))
    return keep

def blocks(length,K,mode):
    if mode=='online':
        # Refit at a block's start, use only observed past; first block retains
        # the original prefix-sized context and queries begin at the common320.
        return [(int(v[0]),int(v[-1])+1) for v in np.array_split(np.arange(FIRST,length),K) if len(v)]
    return [(int(v[0]),int(v[-1])+1) for v in np.array_split(np.arange(32,length),K) if len(v)]

def design(model,z,node,pa,t,period=0,long_parent=False):
    own=torch.stack([z[(t-l).clamp_min(0),node] for l in range(1,33)],-1)
    columns=[torch.ones_like(own[:,:1]),own[:,:8],model.own(own)]
    if period:
        for lag in [period,2*period]:
            valid=t>=lag;columns.extend([(z[(t-lag).clamp_min(0),node]*valid)[:,None],valid.to(z.dtype)[:,None]])
    parent_start=sum(v.shape[-1] for v in columns)
    # Distributed-lag summaries are appended after the frozen nine-lag encoder;
    # exp weights are not standardized away (the summary itself is standardized).
    for p in pa:
        h=torch.stack([z[t-l,p] for l in range(9)],-1)
        columns.extend([h,model.parent(h)])
        if long_parent:
            for low,high in [(9,16),(17,32)]:
                l=torch.arange(low,high+1,device=z.device)
                w=torch.exp(-(l-low)/8.);w=w/w.sum()
                columns.append((z[t[:,None]-l,p]*w).sum(-1,keepdim=True))
    return torch.cat(columns,-1),parent_start

@torch.inference_mode()
def predict(model,values,graph,config):
    assert values.ndim==2 and values.dtype==torch.float64
    first=config.get('first',FIRST)
    assert first in (32,FIRST) and (first==FIRST or config['mode']=='crossfit')
    T,C=values.shape;device=values.device;t=torch.arange(32,T,device=device)
    residual=torch.zeros_like(values);variance=torch.ones_like(values);baseline=values.clone();prediction=values.clone()
    p=periods(values.cpu().numpy()) if config['periodic'] else [0]*C
    counts=[];factors=feature_rows=0
    for node in range(C):
        pa=torch.where(graph[:,node]>0)[0].tolist()
        lags=list(range(1,33))+([p[node],2*p[node]] if p[node] else [])
        for lo,hi in blocks(T,config['K'],config['mode']):
            if hi<=first:continue
            m=context_mask(T,lo,hi,config['gap'],lags,config['mode'])
            assert m.sum()>=16,('insufficient purged context',T,node,lo,hi,int(m.sum()))
            tr=torch.as_tensor(m,device=device);query=(t>=max(lo,first))&(t<hi)
            norm=config['normalization']
            # For strict fold mode neither held-out labels nor held-out raw
            # values enter normalization. Context may still contain anomalies.
            norm_idx=t[tr] if norm=='fold' else torch.arange(256 if norm=='prefix' else T,device=device)
            center=values[norm_idx].median(0).values
            scale=(values[norm_idx]-center).abs().median(0).values*1.4826
            scale=torch.where(scale<1e-3,torch.ones_like(scale),scale)
            z=(values-center)/scale
            raw,parent_start=design(model,z,node,pa,t,p[node],config['long_parent'])
            ctx=raw[tr];fm=ctx.mean(0);fs=ctx.std(0,unbiased=False).clamp_min(1e-3)
            fm[0]=0;fs[0]=1;f=(raw-fm)/fs;cf=f[tr];cy=z[t[tr],node]
            w=torch.ones_like(cy);eye=torch.eye(f.shape[-1],device=device,dtype=f.dtype)
            for _ in range(3):
                gram=eye+cf.T@(w[:,None]*cf);chol=torch.linalg.cholesky(gram)
                beta=torch.cholesky_solve((cf.T@(w*cy))[:,None],chol)[:,0]
                e=cy-cf@beta;noise=((w*e.square()).sum()+beta.square().sum()+1)/(w.sum()+2)
                noise=noise.clamp_min(1e-4)
                nu=config['nu'];w=torch.ones_like(w) if nu is None else (nu+1)/(nu+e.square()/noise)
            q=f[query];mu=q@beta;inv=torch.cholesky_solve(q.T,chol).T
            residual[t[query],node]=z[t[query],node]-mu
            prediction[t[query],node]=center[node]+scale[node]*mu
            variance[t[query],node]=noise*(1+(q*inv).sum(-1))
            # Parent component removed, retaining own-history and intercept;
            # this is an estimate, NOT the generator's true exogenous baseline.
            parent=q[:,parent_start:]@beta[parent_start:] if pa else torch.zeros_like(mu)
            baseline[t[query],node]=values[t[query],node]-scale[node]*parent
            counts.append(dict(node=node,lo=lo,hi=hi,context_points=int(m.sum()),dimension=f.shape[-1]))
            factors+=3;feature_rows+=T-32
    assert torch.isfinite(residual).all() and torch.isfinite(variance).all() and (variance>0).all()
    return dict(residual=residual[first:],variance=variance[first:],baseline=baseline[first:],prediction=prediction[first:],
                audit=dict(periods=p,folds=counts,factorizations=factors,feature_node_rows=feature_rows,
                           future_context=config['mode']!='online',global_weight_updates=0))
