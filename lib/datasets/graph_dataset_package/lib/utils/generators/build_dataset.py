from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from .config import DatasetConfig
from .families import FAMILY_BUILDERS
from .features import build_node_features
from .io import save_generated_dataset
from .splits import make_splits


def _graph_to_edge_index(g: nx.Graph) -> torch.Tensor:
    pyg_graph = from_networkx(g)
    return pyg_graph.edge_index


def _make_graph_label(family_spec, family_name: str) -> int:
    if family_spec.label is not None:
        return int(family_spec.label)
    return 0


def _node_mask_from_membership(num_nodes: int, members: List[int]) -> torch.Tensor:
    y = torch.zeros(num_nodes, dtype=torch.long)
    if members:
        y[torch.tensor(members, dtype=torch.long)] = 1
    return y


def _edge_mask_from_membership(edge_index: torch.Tensor, member_edges: set[tuple[int, int]]) -> torch.Tensor:
    e = edge_index.size(1)
    y = torch.zeros(e, dtype=torch.long)
    for idx in range(e):
        u = int(edge_index[0, idx])
        v = int(edge_index[1, idx])
        if (u, v) in member_edges or (v, u) in member_edges:
            y[idx] = 1
    return y


def _build_single_graph(family_spec, cfg: DatasetConfig, rng: np.random.Generator, graph_seed: int):
    builder = FAMILY_BUILDERS[family_spec.name]
    extra = None
    out = builder(rng, family_spec.n_min, family_spec.n_max, family_spec.params)
    if isinstance(out, tuple):
        g, extra = out
    else:
        g = out

    g = nx.convert_node_labels_to_integers(g)
    num_nodes = g.number_of_nodes()
    x = build_node_features(g, cfg.feature_regime, cfg.random_feature_dim, seed=graph_seed)
    edge_index = _graph_to_edge_index(g)

    metadata = {
        'family': family_spec.name,
        'num_nodes': num_nodes,
        'num_edges': int(g.number_of_edges()),
        'graph_seed': graph_seed,
    }

    if cfg.task_type == 'graph':
        y = torch.tensor(_make_graph_label(family_spec, family_spec.name), dtype=torch.long)

    elif cfg.task_type == 'node':
        if family_spec.name in {'planted_clique', 'hole_rich'}:
            members = extra or []
            y = _node_mask_from_membership(num_nodes, members)
        else:
            deg = np.array([g.degree[v] for v in range(num_nodes)])
            threshold = np.median(deg)
            y = torch.tensor((deg > threshold).astype(np.int64))
        metadata['positive_nodes'] = int(y.sum().item())

    elif cfg.task_type == 'edge':
        if family_spec.name == 'hole_rich' and extra:
            cycle_edges = set()
            cyc = list(extra)
            for i in range(len(cyc)):
                cycle_edges.add((cyc[i], cyc[(i + 1) % len(cyc)]))
            y = _edge_mask_from_membership(edge_index, cycle_edges)
        else:
            y = torch.zeros(edge_index.size(1), dtype=torch.long)
        metadata['positive_edges'] = int(y.sum().item())

    else:
        raise ValueError(f'Unsupported task type: {cfg.task_type}')

    graph = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
    return graph, metadata


def build_dataset(cfg: DatasetConfig):
    rng = np.random.default_rng(cfg.seed)
    graphs: List[Data] = []
    metadata: List[Dict] = []

    for family_spec in cfg.families:
        for _ in range(family_spec.num_graphs):
            graph_seed = int(rng.integers(1_000_000_000))
            graph, meta = _build_single_graph(family_spec, cfg, rng, graph_seed)
            graphs.append(graph)
            metadata.append(meta)

    train_ids, val_ids, test_ids = make_splits(graphs, metadata, cfg.split, cfg.seed)
    return graphs, train_ids, val_ids, test_ids, metadata


def build_and_save_dataset(raw_root: str, cfg: DatasetConfig):
    graphs, train_ids, val_ids, test_ids, metadata = build_dataset(cfg)
    save_generated_dataset(
        raw_root=raw_root,
        dataset_name=cfg.name,
        graphs=graphs,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        metadata=metadata,
        config=cfg.to_dict(),
    )
    return graphs, train_ids, val_ids, test_ids, metadata
