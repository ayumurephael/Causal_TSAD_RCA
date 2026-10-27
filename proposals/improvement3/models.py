"""Matched-capacity query routing, statistic supervision, protected TSAD increment.

All trainable modules consume whitelisted cached observed-derived tensors.
Targets are passed only to loss(), never to forward(). RCA is private to causal
heads; TimeRCD is never given a root head or evaluated for RCA.
"""
import math
import torch
import torch.nn.functional as F
from .evidence import mix,weights,temporal_features,pooled

ARMS=['context_nll','query_nll','query_shuffled_nll','query_rank',
      'stat_pair','stat_temporal','stat_typed','decompose_learned','generic_calibrator',
      'protected','raw_adapt','concat_adapt']

class CandidateSelector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(20,32),torch.nn.Tanh(),torch.nn.Linear(32,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,z,mode):
        x=z['features'].clone()
        if mode=='context_nll':x[...,12:18]=0
        if mode=='query_shuffled_nll':
            # Fixed response-free temporal derangement; same at train/eval.
            x[...,12:18]=x[...,12:18].roll(max(1,len(x)//2),0)
        return self.net(x).squeeze(-1).masked_fill(~z['valid'],-torch.inf).log_softmax(-1)

class CandidateMethod(torch.nn.Module):
    def __init__(self,arm,seed):
        super().__init__();torch.manual_seed(seed);assert arm in ARMS;self.arm=arm
        self.selector=CandidateSelector()
        self.stat=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.Tanh(),torch.nn.Linear(32,1))
        self.root_head=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.Tanh(),torch.nn.Linear(32,4))
        self.cal=torch.nn.Sequential(torch.nn.Linear(6,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
        for module in [self.stat,self.root_head,self.cal]:
            torch.nn.init.zeros_(module[-1].weight);torch.nn.init.zeros_(module[-1].bias)
        for name,module in [('selector',self.selector),('stat',self.stat),('root_head',self.root_head),('cal',self.cal)]:
            enabled=(name=='selector' and arm.startswith(('context','query','stat_'))) or (name=='stat' and arm.startswith('stat_')) or (name=='root_head' and arm=='stat_typed') or (name=='cal' and arm in ['protected','raw_adapt','concat_adapt','decompose_learned','generic_calibrator'])
            module.requires_grad_(enabled)
        self.double()
    def forward(self,z):
        arm=self.arm
        lw=self.selector(z,arm) if arm.startswith(('context','query','stat_')) else weights(z,'uniform' if arm in ['decompose_learned','generic_calibrator'] else 'full')
        m=mix(z,lw);node=m['e'];score=pooled(node);types=None
        if arm.startswith('stat_'):
            x,_,_=temporal_features(z,lw);node=node+self.stat(x).squeeze(-1)
            score=pooled(node)
            if arm=='stat_typed':
                types=self.root_head(x.detach())+m['e'].detach()[...,None]
                # Private RCA evidence does not feed detection loss/readout.
                root=torch.logsumexp(types,-1)-math.log(4)
            else:root=node
        else:root=node
        if arm in ['decompose_learned','generic_calibrator']:
            if arm=='decompose_learned':
                x=torch.stack([m['A']/3,m['D']/3,node,m['precision'].log(),m['mean'],torch.zeros_like(node)],-1)
            else:x=torch.stack([node,node.square(),node.abs(),torch.zeros_like(node),torch.zeros_like(node),torch.zeros_like(node)],-1)
            x=x.sign()*torch.log1p(x.abs());node=node+2*self.cal(x).squeeze(-1).tanh();root=node;score=pooled(node)
        if arm in ['protected','raw_adapt','concat_adapt']:
            raw=z['concat_logit'] if arm=='concat_adapt' else z['raw_logit']
            sf,_,_=temporal_features(z,lw)
            features=torch.stack([raw,pooled(m['A']/3),pooled(m['D']/3),score,sf[...,1::4].amax((1,2)),z['features'][...,12].mean((1,2))],-1)
            if arm=='raw_adapt':features[:,1:]=0
            features=features.sign()*torch.log1p(features.abs())
            score=raw+2*self.cal(features).squeeze(-1).tanh()
            # An independent causal localization may be composed later; the
            # frozen TimeRCD itself is not used as a root scorer.
            root=None
        return dict(score=score,root=root,types=types,lw=lw,mean=m['mean'],variance=1/m['precision'])

def balanced_bce(score,label):
    positive=label>0;loss=F.binary_cross_entropy_with_logits(score,label.to(score.dtype),reduction='none')
    if positive.any() and (~positive).any():return .5*(loss[positive].mean()+loss[~positive].mean())
    return loss.mean()

def loss(model,normal,fault,target,label,root,fault_kind):
    pn,pf=model(normal),model(fault);active=label>0
    if model.arm in ['context_nll','query_nll','query_shuffled_nll','query_rank']:
        def nll(z,out,y):
            v=z['variance'];lp=-.5*(math.log(2*math.pi)+v.log()+(y[...,None]-z['mu']).square()/v)
            return -torch.logsumexp(lp+out['lw'],-1).mean()
        nl=.5*(nll(normal,pn,normal['response'])+nll(fault,pf,target))
        rank=F.softplus(pn['score'][active]-pf['score'][active]).mean()
        return nl+(rank if model.arm=='query_rank' else 0),dict(nll=float(nl.detach()),pair_rank=float(rank.detach()))
    pair=F.softplus(pn['score'][active]-pf['score'][active]).mean()
    outside=~active
    within=F.softplus(pf['score'][outside].mean()-pf['score'][active]).mean() if outside.any() else pair*0
    base=.5*(balanced_bce(pn['score'],torch.zeros_like(label))+balanced_bce(pf['score'],label))
    # stat_pair intentionally omits within-case supervision: a case-only gain
    # should not be misreported as onset localization.
    total=base+pair+(0 if model.arm=='stat_pair' else within)
    rca=base*0;regularizer=base*0
    if model.arm.startswith('stat_'):
        v=fault['variance'];lp=-.5*(math.log(2*math.pi)+v.log()+(target[...,None]-fault['mu']).square()/v)
        regularizer=-torch.logsumexp(lp+pf['lw'],-1).mean()
        total=total+.05*regularizer
    if model.arm=='stat_typed':
        type_index={'relation':0,'short':0,'long':0,'physical':1,'edge':2}[fault_kind]
        flat=pf['types'][active].reshape(-1,pf['types'].shape[1]*4)
        rca=F.cross_entropy(flat,torch.full((len(flat),),int(root)*4+type_index,device=flat.device,dtype=torch.long))
        # No-explanation class is trained on normal source cases. It is not an
        # identifiability certificate; confidence/coverage are evaluated.
        normal_types=pn['types'];normal_logits=torch.logsumexp(normal_types,1)
        rca=rca+F.cross_entropy(normal_logits,torch.full((len(normal_logits),),3,device=flat.device,dtype=torch.long))
        total=total+rca
    return total,dict(bce=float(base.detach()),pair_rank=float(pair.detach()),within_rank=float(within.detach()),root_type_CE=float(rca.detach()),normal_nll_regularizer=float(regularizer.detach()))
