"""Observational vs explanation disagreement, scan and structured-edge evidence.

No chi-square calibration is asserted. Correlated, estimated, scanned evidence
is calibrated empirically on independent source NORMAL cases in the runner.
"""
import math
import torch
import torch.nn.functional as F
WINDOWS=(1,8,32,128)

def mix(z, lw):
    p=lw.exp();inv=p/z['variance'];a=inv.sum(-1)
    mean=(inv*z['mu']).sum(-1)/a.clamp_min(1e-12)
    A=a*(z['response']-mean).square()
    D=(inv*(z['mu']-mean[...,None]).square()).sum(-1)
    e=(A+D)/3-.5*math.log(3)
    return dict(A=A,D=D,e=e,mean=mean,precision=a)

def weights(z, mode):
    valid=z['valid'];logits=torch.zeros_like(z['mu'])
    if mode=='full':logits[...,1:]=-torch.inf
    elif mode=='min_noise':
        v=z['features'][...,2].masked_fill(~valid,torch.inf);i=v.argmin(-1,keepdim=True)
        logits.fill_(-torch.inf).scatter_(-1,i,0)
    elif mode=='consistency':logits=-z['features'][...,18]
    elif mode!='uniform':raise ValueError(mode)
    return logits.masked_fill(~valid,-torch.inf).log_softmax(-1)

def window_sum(x,w):
    # Trailing observed-only windows; no duplicated padding samples.
    cs=torch.cat([torch.zeros_like(x[:1]),x.cumsum(0)],0)
    t=torch.arange(len(x),device=x.device)+1
    return cs[t]-cs[(t-w).clamp_min(0)]

def temporal_features(z, lw):
    m=mix(z,lw);e=m['e'];r=z['response']-m['mean'];v=1/m['precision']
    edges=z['edges'];C=e.shape[-1]
    cols=[];edge_scales=[]
    # Full-candidate edge contributions, kept fixed across selector controls:
    # avoids disguising a changed edge decomposition as selector-only effect.
    g=z['contribution'][...,0]
    for w in WINDOWS:
        count=torch.minimum(torch.arange(1,len(e)+1,device=e.device),torch.as_tensor(w,device=e.device)).to(e.dtype)
        energy=window_sum(e,w)/count[:,None].sqrt()
        inbound=torch.zeros_like(e);outbound=torch.zeros_like(e);ib_count=torch.zeros(C,device=e.device,dtype=e.dtype);ob_count=ib_count.clone()
        if len(edges):
            num=window_sum(r[:,edges[:,1]]*g/v[:,edges[:,1]],w)
            den=window_sum(g.square()/v[:,edges[:,1]],w).clamp_min(1e-8)
            edge=.5*num.square()/den
            inbound.index_add_(1,edges[:,1],edge);outbound.index_add_(1,edges[:,0],edge)
            ib_count.index_add_(0,edges[:,1],torch.ones(len(edges),device=e.device,dtype=e.dtype));ob_count.index_add_(0,edges[:,0],torch.ones(len(edges),device=e.device,dtype=e.dtype))
            edge_scales.append(edge)
        else:edge_scales.append(torch.zeros((len(e),0),device=e.device,dtype=e.dtype))
        # Degree-normalized and raw edge mass are both available for controls.
        cols.extend([energy,inbound/ib_count.clamp_min(1),outbound/ob_count.clamp_min(1),inbound+outbound])
    x=torch.stack(cols,-1)
    x=x.sign()*torch.log1p(x.abs())
    return x,edge_scales,m

def pooled(node):return torch.logsumexp(node,-1)-math.log(node.shape[-1])

def frozen(z):
    out={};roots={};details={}
    for mode in ['full','uniform','min_noise','consistency']:
        lw=weights(z,mode);m=mix(z,lw);out[mode]=pooled(m['e']);roots[mode]=m['e']
        if mode=='uniform':
            for name,val in [('deviation',m['A']/3),('disagreement',m['D']/3)]:out[name]=pooled(val);roots[name]=val
            direct=(((z['response'][...,None]-z['mu']).square()/(3*z['variance'])-.5*math.log(3))*lw.exp()).sum(-1)
            details['decomposition_max_error']=float((direct-m['e']).abs().max())
    e=((z['response'][...,None]-z['mu']).square()/(3*z['variance'])-.5*math.log(3)).masked_fill(~z['valid'],-torch.inf).max(-1).values
    out['max_candidate']=pooled(e);roots['max_candidate']=e
    p=weights(z,'uniform').exp();same=dict(z)
    same['variance']=(p*z['variance']).sum(-1,keepdim=True).expand_as(z['variance'])
    sm=mix(same,weights(same,'uniform'));out['same_variance']=pooled(sm['e']);roots['same_variance']=sm['e']
    x,edges,m=temporal_features(z,weights(z,'full'))
    scans=[];node_scans=[];root_scans=[];shift_scans=[];typed_scans=[];typed_roots=[]
    for j,w in enumerate(WINDOWS):
        count=torch.minimum(torch.arange(1,len(x)+1,device=x.device),torch.as_tensor(w,device=x.device)).to(x.dtype)
        ne=window_sum(m['e'],w)/count[:,None].sqrt()
        node_scans.append(ne-math.log(max(1,len(x)/w))-math.log(len(WINDOWS)))
        edge=edges[j]
        out['node_w'+str(w)]=pooled(ne);roots['node_w'+str(w)]=ne
        if edge.shape[-1]:
            s=torch.logsumexp(edge,-1)-math.log(edge.shape[-1]);cover=torch.zeros_like(ne)
            cover.index_add_(1,z['edges'][:,0],edge);cover.index_add_(1,z['edges'][:,1],edge)
        else:s=pooled(ne);cover=ne
        out['edge_w'+str(w)]=s;roots['edge_w'+str(w)]=cover
        penalty=math.log(max(1,len(x)/w))+math.log(len(WINDOWS))
        scans.append(s-penalty);root_scans.append(cover-penalty)
        # Same predictor/residual/variance, wrong timing of fitted edge signal:
        # this tests directional evidence, not a whole different normal model.
        if len(z['edges']):
            g=z['contribution'][...,0].roll(max(1,len(x)//2),0);c=z['edges'][:,1]
            r=z['response']-m['mean'];v=1/m['precision']
            a=window_sum(r[:,c]*g/v[:,c],w);b=window_sum(g.square()/v[:,c],w).clamp_min(1e-8)
            se=.5*a.square()/b;shift_scans.append(torch.logsumexp(se,-1)-math.log(se.shape[-1])-penalty)
        else:shift_scans.append(s-penalty)
        # Explicit, falsifiable composite signatures. These are heuristic
        # scores, not normalized likelihoods or calibrated Bayes posteriors.
        own,inc,outg=x[...,4*j],x[...,4*j+1],x[...,4*j+2]
        hyp=torch.stack([own+.5*(inc+outg),own-.5*(inc+outg),inc+.25*own-.5*outg],-1)
        node=torch.logsumexp(hyp,-1)-math.log(3)
        typed_scans.append(pooled(node)-penalty);typed_roots.append(node-penalty)
    out['edge_scan']=torch.stack(scans,-1).max(-1).values;roots['edge_scan']=torch.stack(root_scans,-1).max(-1).values
    ne=torch.stack(node_scans,-1).max(-1).values;out['node_scan']=pooled(ne);roots['node_scan']=ne
    out['edge_shift_scan']=torch.stack(shift_scans,-1).max(-1).values
    out['structured_scan']=torch.stack(typed_scans,-1).max(-1).values;roots['structured_scan']=torch.stack(typed_roots,-1).max(-1).values
    # A diagnostic conflict score; NOT a formal identifiability decision. Its
    # operational abstention cutoff is separately fitted on normal sources.
    out['context_conflict']=z['features'][...,18].mean((1,2))
    out['raw']=z['raw_logit'];out['concat']=z['concat_logit']
    degree=torch.zeros(z['response'].shape[-1],device=z['response'].device,dtype=z['response'].dtype)
    if len(z['edges']):
        degree.index_add_(0,z['edges'][:,0],torch.ones(len(z['edges']),device=degree.device,dtype=degree.dtype))
        degree.index_add_(0,z['edges'][:,1],torch.ones(len(z['edges']),device=degree.device,dtype=degree.dtype))
    out['full_degree_RCA']=out['full'];roots['full_degree_RCA']=degree[None].expand_as(z['response'])
    # Explicitly no TimeRCD RCA entries.
    return out,roots,details
