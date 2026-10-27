"""Shared causal-role neural basis with a forward-inferred robust ridge last layer.

Independent ALPaCA-inspired probe, not the author implementation or exact
Bayesian regression with known noise. Prefix coefficients are latent context
estimates; global network weights never update on evaluation cases.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

FIRST=320
class CausalMetaBasis(nn.Module):
    def __init__(self,seed=348):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.own=nn.Sequential(nn.Linear(32,32),nn.Tanh(),nn.Linear(32,8),nn.Tanh())
            self.parent=nn.Sequential(nn.Linear(9,16),nn.Tanh(),nn.Linear(16,4),nn.Tanh())
        self.log_temperature=nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.bias=nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.double()

    def design(self,z,node,parents):
        # z has [view,time,channel]. Each neural basis is local to one sensor.
        t=torch.arange(32,z.shape[1],device=z.device)
        own=torch.stack([z[:,t-lag,node] for lag in range(1,33)],-1)
        columns=[torch.ones_like(own[...,:1]),own[...,:8],self.own(own)]
        for parent in parents:
            history=torch.stack([z[:,t-lag,parent] for lag in range(9)],-1)
            columns.extend([history,self.parent(history)])
        return torch.cat(columns,-1)

    def forward(self,values,graph,detach_detection=False):
        assert values.dtype==torch.float64 and values.ndim==3
        assert values.shape[1]>FIRST and graph.shape==(values.shape[2],)*2
        center=values[:,:256].median(1).values
        scale=(values[:,:256]-center[:,None]).abs().median(1).values*1.4826
        scale=torch.where(scale<1e-3,torch.ones_like(scale),scale)
        z=(values-center[:,None])/scale[:,None]
        means=[];variances=[];residuals=[];conditionings=0
        for node in range(z.shape[2]):
            parents=torch.where(graph[:,node]>0)[0].tolist()
            raw=self.design(z,node,parents)
            ctx=raw[:,:224]
            fm=ctx.mean(1);fs=ctx.std(1,unbiased=False).clamp_min(1e-3)
            # The intercept is never normalized away.
            mask=torch.arange(raw.shape[-1],device=raw.device)!=0
            fm=fm*mask;fs=torch.where(mask,fs,torch.ones_like(fs))
            f=(raw-fm[:,None])/fs[:,None];cf=f[:,:224];cy=z[:,32:256,node]
            w=torch.ones_like(cy);eye=torch.eye(cf.shape[-1],dtype=cf.dtype,device=cf.device)
            for _ in range(3):
                gram=eye+cf.transpose(1,2)@(w[...,None]*cf)
                chol=torch.linalg.cholesky(gram)
                beta=torch.cholesky_solve((cf.transpose(1,2)@(w*cy)[...,None]),chol).squeeze(-1)
                error=cy-(cf*beta[:,None]).sum(-1)
                noise=((w*error.square()).sum(1)+beta.square().sum(1)+1)/(w.sum(1)+2)
                noise=noise.clamp_min(1e-4);w=5/(4+error.square()/noise[:,None])
            query=f[:,FIRST-32:]
            mu=(query*beta[:,None]).sum(-1)
            invf=torch.cholesky_solve(query.transpose(1,2),chol).transpose(1,2)
            variance=noise[:,None]*(1+(query*invf).sum(-1))
            residual=z[:,FIRST:,node]-mu
            means.append(mu);variances.append(variance);residuals.append(residual)
            conditionings+=3*len(values)
        mu=torch.stack(means,-1);var=torch.stack(variances,-1);res=torch.stack(residuals,-1)
        normal_nll=.5*(var.log()+res.square()/var+math.log(2*math.pi))
        # Declared diagonal variance-inflation evidence, NOT the C04h joint kernel.
        e=.5*((2/3)*res.square()/var-math.log(3))
        e=F.avg_pool1d(F.pad(e.transpose(1,2),(15,0),mode='replicate'),16,stride=1).transpose(1,2)
        if detach_detection:e=e.detach()
        temperature=self.log_temperature.exp().clamp(.1,10)
        score=temperature*(torch.logsumexp(e/temperature,-1)-math.log(e.shape[-1]))+self.bias
        return dict(score=score,root=e.softmax(-1),normal_nll=normal_nll,
                    residual=res,variance=var,mean=mu,context_factorizations=conditionings)

def relation_loss(score):
    """Complete-column matched quartet, two normal/two broken relations."""
    assert score.shape[0]==4
    y=torch.tensor([0.,0.,1.,1.],device=score.device,dtype=score.dtype)[:,None].expand_as(score)
    return F.binary_cross_entropy_with_logits(score,y)
