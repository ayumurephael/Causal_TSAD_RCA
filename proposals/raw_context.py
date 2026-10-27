"""Raw parent-child context, before teacher compression. No targets in predictor API."""
import math,hashlib
import numpy as np
import torch
from torch import nn

ARMS=['true_none','true_mean','true_attention','degree_attention','wrong_attention','degree_mean','wrong_mean']
OWN_LAGS=(1,2,4,8,16,32,64)
PARENT_LAGS=(0,1,2,4,8,16,32,64)
CONTEXT_TIMES=tuple(range(64,256,4))
FIRST_QUERY=320

def prepare(values,valid=None):
 """Prefix statistics are channel-local and end before every scored query."""
 if valid is None:valid=torch.ones(len(values),dtype=torch.bool,device=values.device)
 values=torch.where(valid[:,None],values,torch.zeros_like(values))
 prefix=valid[:256,None];count=prefix.sum(0).clamp_min(1)
 center=(values[:256]*prefix).sum(0)/count
 scale=(((values[:256]-center).square()*prefix).sum(0)/count).sqrt()
 scale=torch.where(scale<1e-3,torch.ones_like(scale),scale)
 z=(values-center)/scale
 times=torch.arange(FIRST_QUERY,len(z),device=z.device)
 ctx=torch.tensor(CONTEXT_TIMES,device=z.device)
 def lagged(t,lags):
  indices=t[:,None]-torch.tensor(lags,device=z.device)[None]
  return z[indices].permute(0,2,1)
 inputs=dict(own=lagged(times,OWN_LAGS),parent=lagged(times,PARENT_LAGS),
  context_own=lagged(ctx,OWN_LAGS),context_parent=lagged(ctx,PARENT_LAGS),context_y=z[ctx],context_valid=valid[ctx])
 return inputs,z[times],times,scale

def wrong_graph(graph,key):
 """Fixed degree-matched predictive control; may contain descendants/cycles."""
 out=torch.zeros_like(graph);c=graph.shape[0]
 for child in range(c):
  degree=int(graph[:,child].sum().item())
  choices=[p for p in range(c) if p!=child]
  choices.sort(key=lambda p:hashlib.sha256((key+'/'+str(child)+'/'+str(p)).encode()).hexdigest())
  parents=choices[:degree]
  old=set(torch.where(graph[:,child]>0)[0].cpu().tolist())
  if degree and degree<c-1 and set(parents)==old:
   parents[-1]=next(p for p in choices if p not in old)
  out[parents,child]=1
 return out

def signed_log(x):return torch.sign(x)*torch.log1p(x.abs())

class RawContextPredictor(nn.Module):
 def __init__(self,width=32):
  super().__init__();self.width=width;self.attention_forward_FLOPs=0
  self.edge_value=nn.Sequential(nn.Linear(16,width),nn.GELU(),nn.Linear(width,width))
  self.edge_key=nn.Sequential(nn.Linear(15,width),nn.Tanh())
  self.self_value=nn.Sequential(nn.Linear(8,width),nn.GELU(),nn.Linear(width,width))
  self.self_key=nn.Sequential(nn.Linear(7,width),nn.Tanh())
  self.edge_output=nn.Sequential(nn.Linear(15+width,width),nn.GELU(),nn.Linear(width,width))
  self.decoder=nn.Sequential(nn.Linear(7+2*width+1,64),nn.GELU(),nn.Linear(64,64),nn.GELU(),nn.Linear(64,2))

 def forward(self,own,parent,context_own,context_parent,context_y,context_valid,graph,arm):
  assert arm in ARMS and torch.all(graph.diag()==0)
  c=graph.shape[0];mode='none' if arm.endswith('none') else 'mean' if arm.endswith('mean') else 'attention'
  self.attention_forward_FLOPs+=4*c*(c+1)*own.shape[0]*context_own.shape[0]*self.width
  if arm.startswith('degree'):
   parent=parent*0;context_parent=context_parent*0
  def edge(o,p):
   a=o.permute(1,0,2)[:,None].expand(c,c,-1,-1)
   b=p.permute(1,0,2)[None].expand(c,c,-1,-1)
   return signed_log(torch.cat([a,b],-1))
  keys=edge(context_own,context_parent);queries=edge(own,parent)
  response=signed_log(context_y.T[:,None,:,None].expand(c,c,-1,-1))
  value=self.edge_value(torch.cat([keys,response],-1));key=self.edge_key(keys);query=self.edge_key(queries)
  edge_logits=torch.einsum('ijqd,ijkd->ijqk',query,key)/math.sqrt(self.width)
  self_k=self.self_key(signed_log(context_own.permute(1,0,2)));self_q=self.self_key(signed_log(own.permute(1,0,2)))
  self_v=self.self_value(signed_log(torch.cat([context_own,context_y[...,None]],-1)).permute(1,0,2))
  self_logits=torch.einsum('cqd,ckd->cqk',self_q,self_k)/math.sqrt(self.width)
  def weights(logits):
   masked=logits.masked_fill(~context_valid,-1e4)
   w=masked.softmax(-1)*context_valid
   w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
   if mode=='mean':w=w*0+context_valid/context_valid.sum().clamp_min(1)
   return w
  edge_rep=torch.einsum('ijqk,ijkd->ijqd',weights(edge_logits),value)
  self_rep=torch.einsum('cqk,ckd->cqd',weights(self_logits),self_v)
  if mode=='none':edge_rep=edge_rep*0;self_rep=self_rep*0
  edge_rep=self.edge_output(torch.cat([queries,edge_rep],-1))
  # Pool only permitted parents; computations for all pair slots match controls.
  mask=graph.T[:,:,None,None];degree=mask.sum(1)
  pooled=(edge_rep*mask).sum(1)/degree.clamp_min(1)
  degree_feature=torch.log1p(degree).expand(-1,own.shape[0],-1)
  inputs=torch.cat([signed_log(own.permute(1,0,2)),self_rep,pooled,degree_feature],-1)
  mu,lv=self.decoder(inputs).permute(1,0,2).unbind(-1)
  return mu,lv.clamp(-6,6)

 def predict(self,inputs,graph,arm):
  return self(**inputs,graph=graph,arm=arm)

def losses(mu,lv,y):
 sq=(y-mu).square();standardized=sq*torch.exp(-lv)
 return .5*(lv+standardized+math.log(2*math.pi)),sq,torch.log1p(standardized)

def parameter_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
