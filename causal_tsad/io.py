"""Whitelisted model inputs. Metadata and targets never enter the model call."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.shape).encode() + a.dtype.str.encode())
    h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.partial')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def adjacency(graph, channels):
    """Return source,row -> target,column adjacency, verifying the normal DAG."""
    assert graph['num_nodes'] == channels
    assert graph['edge_convention'] == 'source,target'
    a = np.zeros((channels, channels), np.float32)
    edges = np.asarray(graph['edges'], dtype=np.int64).reshape(-1, 2)
    for src, dst in edges:
        assert 0 <= src < channels and 0 <= dst < channels and src != dst
        a[src, dst] = 1
    degree = a.sum(0).astype(int)
    queue = list(np.flatnonzero(degree == 0)); visited = 0
    while queue:
        src = queue.pop(); visited += 1
        for dst in np.flatnonzero(a[src]):
            degree[dst] -= 1
            if degree[dst] == 0: queue.append(int(dst))
    assert visited == channels, 'Normal graph must be a DAG'
    return a


def model_inputs(record, view='anomalous'):
    assert view in ('normal', 'anomalous')
    key = 'normal_time_series' if view == 'normal' else 'time_series'
    x = np.ascontiguousarray(record[key], dtype=np.float32)
    assert x.ndim == 2 and np.isfinite(x).all()
    return dict(values=x, valid=np.ones(len(x), bool), adjacency=adjacency(record['graph'], x.shape[1]))


def graph_control(a, mode, permutation):
    if mode == 'parents': return a.copy()
    if mode == 'none': return np.zeros_like(a)
    if mode == 'full': return np.ones_like(a) - np.eye(len(a), dtype=a.dtype)
    if mode == 'rewired': return a[np.ix_(permutation, permutation)].copy()
    raise ValueError(mode)
