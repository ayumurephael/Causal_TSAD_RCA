"""Per-case point/temporal metrics and detected-event localization.

No concatenated discontinuous cases for VUS. Threshold fitted ONLY on source
normal calibration cases; all misses count against joint detected-root recall.
"""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,auc

def spans(mask):
    d=np.diff(np.r_[0,np.asarray(mask,dtype=np.int8),0])
    # Event bounds also enter differences and comparisons below. Native ints
    # keep the complete event record JSON-safe without changing the metric.
    return [(int(a),int(b)) for a,b in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1))]

def evaluate(score,label,threshold,node=None,root=None):
    s=np.asarray(score);y=np.asarray(label,dtype=np.int8);assert s.shape==y.shape and np.isfinite(s).all()
    out=dict(points=len(y),positive_points=int(y.sum()),prevalence=float(y.mean()),case_mean_score=float(s.mean()),case_max_score=float(s.max()),FPR=float((s>threshold).mean()) if not y.any() else None)
    if not y.any():
        out['false_alarm_events']=len(spans(s>threshold));return out
    p,r,_=precision_recall_curve(y,s)
    out.update(AP=float(average_precision_score(y,s)),PR_AUC=float(auc(r,p)),ROC_AUC=float(roc_auc_score(y,s)) if (~y.astype(bool)).any() else None)
    events=spans(y>0);alarms=spans(s>threshold);used=set();event_results=[]
    for lo,hi in events:
        candidates=[(max(0,min(hi,b)-max(lo,a)),j,a,b) for j,(a,b) in enumerate(alarms) if j not in used]
        best=max(candidates,default=(0,-1,0,0));hit=best[0]>0;rank=None;joint=False
        conditional=None
        if node is not None and root is not None:
            def getrank(a,b):
                x=np.asarray(node[a:b]);x=x-x.max(1,keepdims=True);w=np.exp(x);w/=w.sum(1,keepdims=True)
                return int(np.flatnonzero(np.argsort(-w.mean(0),kind='stable')==root)[0])+1
            conditional=getrank(lo,hi)
            if hit:rank=getrank(best[2],best[3]);joint=rank==1
        if hit:used.add(best[1])
        event_results.append(dict(onset=lo,offset=hi,detected=hit,onset_error=best[2]-lo if hit else None,
                                  conditional_rank=conditional,detected_window_rank=rank,joint_correct=joint))
    out['events']=event_results;out['event_recall']=float(np.mean([e['detected'] for e in event_results]))
    out['unmatched_alarm_events']=len(alarms)-len(used)
    if node is not None:
        out['conditional_Top1']=float(np.mean([e['conditional_rank']==1 for e in event_results]))
        out['conditional_MRR']=float(np.mean([1/e['conditional_rank'] for e in event_results]))
        out['joint_detected_root_recall']=float(np.mean([e['joint_correct'] for e in event_results]))
    return out
