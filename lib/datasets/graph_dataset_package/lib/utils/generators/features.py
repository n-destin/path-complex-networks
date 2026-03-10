from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import torch


def build_node_features(
    g: nx.Graph,
    regime: str = 'ones',
    random_feature_dim: int = 8,
    seed: int = 0,
) -> torch.Tensor:
    nodes = list(g.nodes())
    n = len(nodes)

    if regime == 'ones':
        return torch.ones((n, 1), dtype=torch.float32)

    if regime == 'degree':
        deg = np.array([g.degree[v] for v in nodes], dtype=np.float32).reshape(n, 1)
        max_deg = max(float(deg.max()), 1.0)
        return torch.from_numpy(deg / max_deg)

    if regime == 'degree_triangle':
        deg = np.array([g.degree[v] for v in nodes], dtype=np.float32).reshape(n, 1)
        tri = nx.triangles(g)
        tri_arr = np.array([tri[v] for v in nodes], dtype=np.float32).reshape(n, 1)
        if deg.max() > 0:
            deg = deg / deg.max()
        if tri_arr.max() > 0:
            tri_arr = tri_arr / tri_arr.max()
        return torch.from_numpy(np.concatenate([deg, tri_arr], axis=1))

    if regime == 'random':
        rng = np.random.default_rng(seed)
        feats = rng.normal(size=(n, random_feature_dim)).astype(np.float32)
        return torch.from_numpy(feats)

    raise ValueError(f'Unknown feature regime: {regime}')
