"""Make a local portable plan for the original normal-basis pretraining."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import sha256,atomic_json
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-regenerated-source',action='store_true',help='Explicitly register a separate reproduction if rebuilt data bytes differ');a=p.parse_args();assert not a.output.exists()
    cfg=json.loads((ROOT/'configs/basis_training.json').read_text());actual=sha256(a.source/'manifest.json')
    if actual!=cfg['source_SHA256']:
        assert a.allow_regenerated_source,'Rebuilt source differs. Inspect, then explicitly use --allow-regenerated-source for a separate reproduction, not original-score verification.'
        old=json.loads((ROOT/'provenance/basis_source_manifest.json').read_text());new=json.loads((a.source/'manifest.json').read_text());expected={r['key']:r for r in old['files']}
        assert len(new['files'])==len(expected)==80 and new['parent_source_SHA256']==old['parent_source_SHA256']
        differences=[]
        from causal_tsad.io import array_sha
        import numpy as np
        for row in new['files']:
            ref=expected[row['key']]
            for k in ['split','C','root','family','graph_SHA256','prefix_SHA256']:assert row[k]==ref[k],(row['key'],k)
            file=a.source/row['path'];assert sha256(file)==row['SHA256']
            with np.load(file) as f:
                assert array_sha(f['cases'])==row['input_cases_SHA256'] and array_sha(f['graph'])==row['graph_SHA256']
            if row['input_cases_SHA256']!=ref['input_cases_SHA256']:differences.append(row['key'])
        cfg.update(original_source_SHA256=cfg['source_SHA256'],source_SHA256=actual,regenerated_source=True,array_hash_different_keys=differences,original_numeric_equality_not_asserted=True)
        cfg['training_files']=[dict(key=r['key'],SHA256=r['SHA256']) for r in new['files'] if r['split']=='train']
    cfg.update(source=str(a.source.resolve()),deadline_unix=time.time()+21600)
    atomic_json(a.output,cfg)
