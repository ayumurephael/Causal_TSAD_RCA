"""Independent SCM source quartets with exact complete univariate class matching."""
import hashlib
import numpy as np

T=1024
FORK=256

def generate(channels,seed,family,fork=FORK):
 rng=np.random.default_rng(seed);c=channels;g=np.zeros((c,c),np.float32)
 for child in range(1,c):
  if rng.random()<.2:continue
  p=rng.choice(child,size=int(rng.integers(1,min(3,child)+1)),replace=False);g[p,child]=1
 if not g.any():g[0,1]=1
 rho=rng.uniform(.1,.65,c);sigma=rng.uniform(.15,.4,c);sigma[g.sum(0)==0]=rng.uniform(.8,1.2,int((g.sum(0)==0).sum()))
 coeff=rng.uniform(.35,.85,(c,c))*rng.choice([-1,1],(c,c))*g/np.sqrt(np.maximum(g.sum(0),1))[None]
 lag=rng.integers(0,9,(c,c));parents=[np.flatnonzero(g[:,i]) for i in range(c)]
 def f(x):return x if family=='linear' else np.tanh(x) if family=='tanh' else np.sin(x)
 def simulate(prefix,start,stop,noise_seed):
  a=np.zeros((T,c),np.float64)
  if prefix is not None:a[:start]=prefix[:start]
  noise=np.random.default_rng(noise_seed).normal(size=(stop-start,c))*sigma
  for t in range(start,stop):
   for i in range(c):
    p=parents[i];delay=t-lag[p,i];good=delay>=0
    a[t,i]=(rho[i]*a[t-1,i] if t else 0)+noise[t-start,i]
    if good.any():a[t,i]+=np.dot(coeff[p[good],i],f(a[delay[good],p[good]]))
  return a
 prefix=simulate(None,0,fork,int(rng.integers(0,2**63)))
 a=simulate(prefix,fork,T,int(rng.integers(0,2**63)));b=simulate(prefix,fork,T,int(rng.integers(0,2**63)))
 root=int(rng.choice(np.flatnonzero(g.sum(0)+g.sum(1)>0)));x=a.copy();y=b.copy();x[fork:,root]=b[fork:,root];y[fork:,root]=a[fork:,root]
 cases=np.stack([a,b,x,y]);labels=np.zeros((4,T),np.int8);labels[2:,fork:]=1
 contaminated=bool(rng.random()<.25)
 if contaminated:
  node=int(rng.integers(c));start=int(rng.integers(64,192));length=int(rng.integers(2,9));amplitude=3*max(float(prefix[:fork,node].std()),.1)
  cases[:,start:start+length,node]+=amplitude;labels[:,start:start+length]=1
 perm=rng.permutation(c);cases=cases[:,:,perm];g=g[perm][:,perm];root=int(np.flatnonzero(perm==root)[0])
 # Each channel's complete normal trajectories are exactly the same multiset as
 # the two faulty trajectories. No pointwise-only or histogram approximation.
 for i in range(c):
  lhs=sorted(hashlib.sha256(z.tobytes()).hexdigest() for z in cases[:2,:,i]);rhs=sorted(hashlib.sha256(z.tobytes()).hexdigest() for z in cases[2:,:,i]);assert lhs==rhs
 assert np.array_equal(cases[:,:fork],np.repeat(cases[:1,:fork],4,0))
 return dict(cases=cases,labels=labels,graph=g,root=root,seed=seed,family=family,prefix_contaminated=contaminated,
  audit_parameters=dict(rho=rho.tolist(),sigma=sigma.tolist(),coefficient_before_permutation=coeff.tolist(),lag_before_permutation=lag.tolist(),new_to_old=perm.tolist()),
  labels_detail=dict(effect=None,anomaly_channel=None,anomaly_time=None),fault='measurement trajectory replacement; descendants retain original physical branch')

def rewire(graph,key):
 """Two-edge swaps preserve both in/out degree; diagnostic graph may have cycles."""
 import torch
 g=graph.detach().cpu().numpy().copy();rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
 for _ in range(256):
  edges=np.argwhere(g>0)
  if len(edges)<2:break
  (a,b),(c,d)=edges[rng.choice(len(edges),2,replace=False)]
  if len({int(a),int(c)})<2 or b==d or a==d or c==b or g[a,d] or g[c,b]:continue
  g[a,b]=g[c,d]=0;g[a,d]=g[c,b]=1
 assert np.array_equal(g.sum(0),graph.cpu().numpy().sum(0)) and np.array_equal(g.sum(1),graph.cpu().numpy().sum(1)) and not np.diag(g).any()
 return torch.from_numpy(g).to(graph.device)
