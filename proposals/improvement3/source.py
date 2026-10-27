"""Independent source systems, unknown onsets, matched relation quartets.

Generator metadata is audit/loss-only. Inference accepts values and normal DAG.
No benchmark arrays are loaded here. Physical/edge labels denote injection,
not thresholded downstream effects. No effect/channel labels are constructed.
"""
import hashlib
import numpy as np

T = 1024
NAMESPACE = 348120000000

def generate(index, split):
    offset = {'train': 0, 'calibration': 10000, 'confirmation': 20000}[split]
    seed = NAMESPACE + offset + index
    rng = np.random.default_rng(seed)
    C = [8, 12, 16, 20][index % 4]
    family = ['linear', 'tanh', 'sin'][index % 3]
    g = np.zeros((C, C), dtype=np.float64)
    for child in range(1, C):
        pa = rng.choice(child, size=int(rng.integers(1, min(child, 3)+1)), replace=False)
        g[pa, child] = 1
    rho = rng.uniform(.1, .65, C)
    sigma = rng.uniform(.15, .4, C)
    sigma[g.sum(0) == 0] = .9
    beta = rng.uniform(.35, .85, (C, C))*rng.choice([-1, 1], (C, C))*g / np.sqrt(np.maximum(g.sum(0), 1))[None]
    lag = rng.integers(0, 9, (C, C))
    root = int(rng.choice(np.flatnonzero(g.sum(0) > 0)))
    parent = int(rng.choice(np.flatnonzero(g[:, root])))
    onset = int(rng.integers(64, 901))
    if split == 'confirmation':
        # Prespecified hard context placements, including late clean context.
        onset = [64, 128, 256, 512, 800, int(rng.integers(64, 901))][index % 6]
    duration = min(int(rng.choice([16, 64, 128, 256, T])), T-onset)
    end = onset+duration
    if split == 'confirmation' and index % 6 == 5:
        onset,end=64,800
    noise_a = rng.normal(size=(T, C))*sigma
    noise_b = rng.normal(size=(T, C))*sigma
    noise_b[:onset] = noise_a[:onset]
    f = (lambda x: x) if family == 'linear' else np.tanh if family == 'tanh' else np.sin
    parents = [np.flatnonzero(g[:, n]) for n in range(C)]
    def simulate(noise, fault=None):
        x = np.zeros_like(noise)
        for t in range(T):
            for n in range(C):
                x[t, n] = noise[t, n] + (rho[n]*x[t-1, n] if t else 0)
                pa = parents[n]; d = t-lag[pa, n]; ok = d >= 0
                co = beta[pa, n].copy()
                if fault == 'edge' and n == root and onset <= t < end:
                    co[pa == parent] *= -1
                if ok.any(): x[t, n] += co[ok] @ f(x[d[ok], pa[ok]])
                if fault == 'physical' and n == root and onset <= t < end:
                    x[t, n] += 2.5*sigma[root]
        return x
    a, b = simulate(noise_a), simulate(noise_b)
    x, y = a.copy(), b.copy()
    x[onset:, root], y[onset:, root] = b[onset:, root], a[onset:, root]
    short, long = a.copy(), a.copy()
    amp = 3*max(float(a[:, root].std()), .1)
    short[onset, root] += amp; long[onset:end, root] += amp
    cases = np.stack([a, b, x, y, short, long, simulate(noise_a, 'physical'), simulate(noise_a, 'edge')])
    labels = np.zeros((8, T), dtype=np.int8); labels[2:, onset:end] = 1
    labels[2:4,onset:] = 1
    labels[4] = 0; labels[4, onset] = 1
    # Relation swaps persist to the end: complete-column matching is exact.
    for n in range(C):
        hs = lambda arr: sorted(hashlib.sha256(v.tobytes()).hexdigest() for v in arr)
        assert hs(cases[:2, :, n]) == hs(cases[2:4, :, n])
    perm = rng.permutation(C); cases = cases[:, :, perm]; g = g[perm][:, perm]
    root = int(np.flatnonzero(perm == root)[0]); parent = int(np.flatnonzero(perm == parent)[0])
    phase = rng.uniform(0, 2*np.pi, C)
    sd = np.maximum(cases[0].std(0), .1)
    smooth = 4*sd[None]*np.sin(2*np.pi*np.arange(T)[:, None]/(.75*T)+phase)
    values = np.concatenate([cases, cases+smooth[None]])
    labels = np.concatenate([labels, labels])
    # Normal source twin is an answer for optional source NLL, never an input.
    normal_index = [0, 1, 0, 1, 0, 0, 0, 0]
    return dict(values=values, labels=labels, graph=g, root=root,
        row=dict(key=f'{split}_{index:03d}', split=split, C=C, family=family, seed=seed,
                 root=root, edge_parent=parent, onset=onset, end=end, root_sampling='non-source nodes; degree-only RCA control required',
                 fault_names=['normal_a', 'normal_b', 'relation_a', 'relation_b', 'short', 'long', 'physical', 'edge'],
                 normal_view=[v+bg*8 for bg in range(2) for v in normal_index]),
        audit=dict(rho=rho.tolist(), sigma=sigma.tolist(), coefficient_before_permutation=beta.tolist(),
                   lag_before_permutation=lag.tolist(), new_to_old=perm.tolist(),
                   labels_detail=dict(effect=None, anomaly_channel=None, anomaly_time=None)))
