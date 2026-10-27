"""Download immutable Release assets, verify SHA256, then safely extract."""
from pathlib import Path
import argparse,json,hashlib,urllib.request,tarfile,zipfile
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def extract(path,dest):
    dest.mkdir(parents=True,exist_ok=True)
    if path.suffix=='.tar':
        with tarfile.open(path) as f:
            for m in f.getmembers():
                assert not m.issym() and not m.islnk() and (m.isdir() or m.isfile())
                assert (dest/m.name).resolve().is_relative_to(dest.resolve())
            f.extractall(dest)
    else:
        with zipfile.ZipFile(path) as f:
            for m in f.infolist():assert (dest/m.filename).resolve().is_relative_to(dest.resolve())
            f.extractall(dest)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asset',choices=['source','basis','scores','all'],default='source');p.add_argument('--extract',action='store_true');a=p.parse_args()
    manifest=json.loads((ROOT/'provenance/release_assets.json').read_text());names={'source':'source_pool_v1.tar','basis':'basis_parent_source_v1.tar','scores':'stat_temporal_scores_v1.zip'}
    for item in manifest['assets']:
        if a.asset!='all' and item['name']!=names[a.asset]:continue
        path=ROOT/'assets'/item['name'];path.parent.mkdir(exist_ok=True)
        if not path.exists():
            url=f"https://github.com/{manifest['repository']}/releases/download/{manifest['tag']}/{item['name']}"
            temp=path.with_suffix(path.suffix+'.partial');urllib.request.urlretrieve(url,temp)
            assert digest(temp)==item['SHA256'];temp.replace(path)
        assert digest(path)==item['SHA256']
        if a.extract:
            target=ROOT/'data'
            if item['name'].startswith('basis_'):target=target/'basis_parent'
            extract(path,target)
        print(dict(asset=item['name'],SHA256_verified=True,extracted=a.extract))
