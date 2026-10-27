"""Builder-only true equations extend old matched source into physical counterparts.

The model is never passed these parameters. No original arrays are modified.
"""
import numpy as np
def physical_counterparts(row,parameters):
 x=row['cases'];perm=np.asarray(parameters['new_to_old']);rho=np.asarray(parameters['rho'])[perm];b=np.asarray(parameters['coefficient_before_permutation'])[perm][:,perm];lag=np.asarray(parameters['lag_before_permutation'])[perm][:,perm];c=x.shape[-1];root=row['root'];order=np.argsort(perm);parents=[np.flatnonzero(b[:,j]) for j in range(c)];f=(lambda v:v) if row['family']=='linear' else np.tanh if row['family']=='tanh' else np.sin;physical=[];measurement=[]
 for base,replacement in [(x[0],x[1]),(x[1],x[0])]:
  out=base.copy();sensor=base.copy();sensor[256:,root]=1.5*replacement[256:,root];measurement.append(sensor)
  for t in range(256,1024):
   for node in order:
    if node==root:out[t,node]=sensor[t,node];continue
    p=parents[node];normal_parent=f(base[t-lag[p,node],p]);changed=f(out[t-lag[p,node],p]);out[t,node]=base[t,node]+rho[node]*(out[t-1,node]-base[t-1,node])+np.dot(b[p,node],changed-normal_parent)
  assert np.isfinite(out).all() and np.array_equal(out[:,root],sensor[:,root]);physical.append(out)
 cases=np.concatenate([x,np.stack(measurement),np.stack(physical)]);assert np.array_equal(cases[4,:,root],cases[6,:,root]) and np.array_equal(cases[5,:,root],cases[7,:,root]);assert np.array_equal(cases[:,:256],np.repeat(cases[:1,:256],8,0));return cases
