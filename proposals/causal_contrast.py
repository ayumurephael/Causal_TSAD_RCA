"""Graph-context discriminator; current observations allowed, prediction branch remains blind."""
import torch
from torch import nn
from proposals.raw_context import RawContextPredictor,prepare,signed_log,losses

class CausalContrast(nn.Module):
 def __init__(self):
  super().__init__();self.predictor=RawContextPredictor();self.graph_aggregate_forward_FLOPs=0
  self.local=nn.Sequential(nn.Linear(12,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
  self.message=nn.Sequential(nn.Linear(98,32),nn.GELU())
  self.detector=nn.Linear(32,1)
  self.root=nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,1))

 def forward(self,values,graph,mode):
  inputs,y,times,scale=prepare(values)
  return self.forward_prepared(inputs,y,times,graph,mode)

 def forward_prepared(self,inputs,y,times,graph,mode):
  arm='degree_mean' if mode=='degree' else 'true_mean'
  mu,lv=self.predictor.predict(inputs,graph,arm);nll,sq,innovation=losses(mu,lv,y)
  node=torch.cat([signed_log((y-mu)*torch.exp(-lv/2))[...,None],innovation[...,None],lv[...,None],signed_log(mu)[...,None],signed_log(y)[...,None],signed_log(inputs['own'])],-1)
  h=self.local(node);degree=graph.sum(0);outdegree=graph.sum(1)
  self.graph_aggregate_forward_FLOPs+=4*h.shape[0]*h.shape[1]**2*h.shape[2]
  parent=torch.einsum('pc,qpd->qcd',graph,h)/degree.clamp_min(1)[None,:,None]
  child=torch.einsum('cp,qpd->qcd',graph,h)/outdegree.clamp_min(1)[None,:,None]
  if mode=='degree':parent=parent*0;child=child*0
  d=torch.stack([torch.log1p(degree),torch.log1p(outdegree)],-1)[None].expand(len(y),-1,-1)
  hidden=self.message(torch.cat([h,parent,child,d],-1))
  det=self.detector(hidden).squeeze(-1).mean(-1)
  # RCA receives features but its gradients cannot alter the detector/predictor.
  root=self.root(hidden.detach()).squeeze(-1)
  return dict(logit=det,root_logits=root,normal_NLL=nll,normal_MSE=sq,ready_times=times)
