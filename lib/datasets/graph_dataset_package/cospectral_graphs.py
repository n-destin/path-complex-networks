import os
import os.path as osp
import json
import math
import random
import pickle
import subprocess
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Iterable, Set, Iterator

import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data


# =========================================================
# Configs
# =========================================================

@dataclass
class FamilySpec:
    """
    Supported family names:
      - er
      - triangle_rich
      - planted_clique
      - hole_rich
      - geng_sparse
      - geng_medium
      - geng_dense
      - geng_mixed
    """
    name: str
    num_graphs: int
    n_min: int
    n_max: int
    label: int
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitConfig:
    scheme: str = "iid"  # iid | size_extrapolation | distribution_shift
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # size extrapolation
    train_max_nodes: Optional[int] = None
    val_max_nodes: Optional[int] = None

    # distribution shift
    train_families: Optional[List[str]] = None
    val_families: Optional[List[str]] = None
    test_families: Optional[List[str]] = None


@dataclass
class DatasetConfig:
    name: str
    task_type: str = "graph"         # graph | node | edge | pair
    feature_regime: str = "degree"     # ones | degree | degree_triangle | random
    seed: int = 0
    families: List[FamilySpec] = field(default_factory=list)
    split: SplitConfig = field(default_factory=SplitConfig)


# =========================================================
# General utils
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def reset_file(path: str):
    parent = osp.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "wb"):
        pass


def nx_to_edge_index(g: nx.Graph) -> torch.Tensor:
    edges = list(g.edges())
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    doubled = []
    for u, v in edges:
        doubled.append((u, v))
        doubled.append((v, u))
    return torch.tensor(doubled, dtype=torch.long).t().contiguous()


def edge_label_tensor_from_dict(g: nx.Graph, edge_label_dict: Dict[Tuple[int, int], int]) -> torch.Tensor:
    labels = []
    for u, v in g.edges():
        key = (u, v) if (u, v) in edge_label_dict else (v, u)
        labels.append(edge_label_dict.get(key, 0))
    return torch.tensor(labels, dtype=torch.long)


def _connect_components(g: nx.Graph) -> nx.Graph:
    if g.number_of_nodes() == 0:
        return g
    if not nx.is_connected(g):
        comps = [list(c) for c in nx.connected_components(g)]
        for i in range(len(comps) - 1):
            g.add_edge(comps[i][0], comps[i + 1][0])
    return nx.convert_node_labels_to_integers(g)


def graph6_from_nx(g: nx.Graph) -> str:
    return nx.to_graph6_bytes(g, header=False).decode().strip()


# =========================================================
# Features
# =========================================================

def make_node_features(g: nx.Graph, regime: str, seed: int = 0) -> torch.Tensor:
    n = g.number_of_nodes()

    if regime == "ones":
        return torch.ones((n, 1), dtype=torch.float32)

    if regime == "degree":
        return torch.tensor([[g.degree(v)] for v in range(n)], dtype=torch.float32)

    if regime == "degree_triangle":
        tri = nx.triangles(g)
        return torch.tensor([[g.degree(v), tri[v]] for v in range(n)], dtype=torch.float32)

    if regime == "random":
        gen = torch.Generator().manual_seed(seed)
        return torch.randn((n, 8), generator=gen, dtype=torch.float32)

    raise ValueError(f"Unknown feature regime: {regime}")


# =========================================================
# Pair-label helpers
# =========================================================

def adj_to_dict(adj: torch.Tensor):
    n = adj.size(0)
    g = defaultdict(list)
    idx = (adj.triu(1) != 0).nonzero(as_tuple=False)
    for i, j in idx.tolist():
        g[i].append(j)
        g[j].append(i)
    return g, n


def bfs_distances(g: dict, src: int, n: int) -> torch.Tensor:
    dist = torch.full((n,), -1, dtype=torch.long)
    dist[src] = 0
    q = [src]
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        du = dist[u].item()
        for v in g[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
    return dist


def all_nodes_shortest_paths(adj: torch.Tensor) -> torch.Tensor:
    g, n = adj_to_dict(adj)
    d = torch.empty((n, n), dtype=torch.long)
    for s in range(n):
        d[s] = bfs_distances(g, s, n)
    return d


def random_walk_probability_matrices(adj: torch.Tensor, T: int) -> torch.Tensor:
    n = adj.size(0)
    adj = adj.to(torch.float32)
    deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    P = adj / deg

    mats = torch.empty((n, n, T + 1), dtype=torch.float32)
    mats[:, :, 0] = torch.eye(n, dtype=torch.float32)

    cur = torch.eye(n, dtype=torch.float32)
    for t in range(1, T + 1):
        cur = cur @ P
        mats[:, :, t] = cur
    return mats


def tunneling_behaviour(prob_mats: torch.Tensor, dists: torch.Tensor) -> torch.Tensor:
    n = dists.size(0)
    pairs = torch.combinations(torch.arange(n), r=2)
    labels = []

    for i, j in pairs.tolist():
        limit = int(dists[i, j].item())
        if limit <= 0:
            labels.append([0, 0, 1])
            continue

        a = prob_mats[i, i, :limit]
        b = prob_mats[j, j, :limit]

        complete = torch.equal(a, b)
        partial = torch.equal(a[:-1], b[:-1]) if limit >= 2 else False

        if complete:
            labels.append([1, 0, 0])
        elif partial:
            labels.append([0, 1, 0])
        else:
            labels.append([0, 0, 1])

    return torch.tensor(labels, dtype=torch.long)


def build_pair_labels_from_graph(g: nx.Graph) -> torch.Tensor:
    adj_np = nx.to_numpy_array(g, dtype=np.int64)
    adj = torch.from_numpy(adj_np)
    dists = all_nodes_shortest_paths(adj)
    Tmax = int(dists.max().item())
    if Tmax < 0:
        num_pairs = math.comb(g.number_of_nodes(), 2)
        return torch.zeros((num_pairs, 3), dtype=torch.long)
    prob_mats = random_walk_probability_matrices(adj, Tmax)
    return tunneling_behaviour(prob_mats, dists)


# =========================================================
# Constructive families
# =========================================================

def build_er_graph(n: int, params: Dict[str, Any]) -> nx.Graph:
    p = params.get("p", min(0.15, max(0.05, 4.0 / max(n, 1))))
    g = nx.erdos_renyi_graph(n=n, p=p)
    return _connect_components(g)


def build_triangle_rich_graph(n: int, params: Dict[str, Any]) -> nx.Graph:
    k = min(params.get("k", 6), n - 1)
    if k % 2 == 1:
        k -= 1
    k = max(k, 2)

    rewire_p = params.get("rewire_p", 0.05)
    g = nx.watts_strogatz_graph(n=n, k=k, p=rewire_p)
    return _connect_components(g)


def build_planted_clique_graph(n: int, params: Dict[str, Any]) -> Tuple[nx.Graph, List[int]]:
    p = params.get("p", min(0.12, max(0.03, 3.5 / max(n, 1))))
    clique_size = min(params.get("clique_size", 5), n)

    g = nx.erdos_renyi_graph(n=n, p=p)
    clique_nodes = list(range(clique_size))
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            g.add_edge(clique_nodes[i], clique_nodes[j])

    g = _connect_components(g)
    return g, clique_nodes


def build_hole_rich_graph(n: int, params: Dict[str, Any]) -> Tuple[nx.Graph, List[int], Dict[Tuple[int, int], int]]:
    num_cycles = params.get("num_cycles", 1)
    min_cycle_len = params.get("min_cycle_len", 6)
    tree_attach_prob = params.get("tree_attach_prob", 0.35)

    g = nx.Graph()
    g.add_nodes_from(range(n))

    used = 0
    cycle_nodes_all = []
    cycle_edge_labels = {}

    for _ in range(num_cycles):
        remaining = n - used
        if remaining < min_cycle_len:
            break

        upper = max(min_cycle_len, min(min_cycle_len + 3, remaining))
        cyc_len = min(random.randint(min_cycle_len, upper), remaining)
        cyc_nodes = list(range(used, used + cyc_len))
        cycle_nodes_all.extend(cyc_nodes)

        for i in range(cyc_len):
            u = cyc_nodes[i]
            v = cyc_nodes[(i + 1) % cyc_len]
            g.add_edge(u, v)
            cycle_edge_labels[(u, v)] = 1

        used += cyc_len

    for v in range(used, n):
        if len(cycle_nodes_all) == 0:
            attach_to = random.randint(0, max(v - 1, 0))
        else:
            attach_to = random.choice(cycle_nodes_all) if random.random() < tree_attach_prob else random.randint(0, max(v - 1, 0))
        g.add_edge(v, attach_to)

    if g.number_of_edges() == 0 and n >= 2:
        for i in range(n - 1):
            g.add_edge(i, i + 1)

    g = _connect_components(g)
    cycle_nodes_all = sorted(set(cycle_nodes_all))
    return g, cycle_nodes_all, cycle_edge_labels


# =========================================================
# geng families
# =========================================================

def edge_bins(n: int):
    mmax = n * (n - 1) // 2
    return {
        "sparse": (n - 1, max(n - 1, int(0.15 * mmax))),
        "medium": (int(0.25 * mmax), int(0.45 * mmax)),
        "dense": (int(0.70 * mmax), mmax),
    }


def geng_stream(n: int, m_lo: int, m_hi: int, res: int, mod: int) -> Iterable[str]:
    args = ["-c", "-q", str(n), f"{m_lo}:{m_hi}", f"{res}/{mod}"]
    p = subprocess.Popen(["geng"] + args, stdout=subprocess.PIPE, text=True)
    try:
        for ln in p.stdout:
            ln = ln.strip()
            if ln:
                yield ln
    finally:
        if p.stdout is not None:
            p.stdout.close()
        p.wait()


def sample_geng_graphs_for_n(
    n: int,
    need: int,
    density_mode: str,
    rng: random.Random,
    mod: int = 2000,
    max_tries: int = 250,
) -> List[nx.Graph]:
    bins = edge_bins(n)
    if density_mode == "sparse":
        selected_bins = [bins["sparse"]]
    elif density_mode == "medium":
        selected_bins = [bins["medium"]]
    elif density_mode == "dense":
        selected_bins = [bins["dense"]]
    elif density_mode == "mixed":
        selected_bins = [bins["sparse"], bins["medium"], bins["dense"]]
    else:
        raise ValueError(f"Unknown geng density mode: {density_mode}")

    if len(selected_bins) == 1:
        per_bin = [need]
    else:
        base = need // len(selected_bins)
        per_bin = [base] * len(selected_bins)
        for i in range(need - base * len(selected_bins)):
            per_bin[i] += 1

    out_graphs = []
    seen_g6: Set[str] = set()

    for (m_lo, m_hi), count in zip(selected_bins, per_bin):
        tries = 0
        while count > 0 and tries < max_tries:
            res = rng.randrange(mod)
            got_any = False
            for g6 in geng_stream(n, m_lo, m_hi, res, mod):
                if g6 in seen_g6:
                    continue
                seen_g6.add(g6)
                g = nx.from_graph6_bytes(g6.encode())
                g = nx.convert_node_labels_to_integers(g)
                out_graphs.append(g)
                count -= 1
                got_any = True
                if count <= 0:
                    break
            tries += 1
            if tries > 20 and not got_any:
                break

    return out_graphs[:need]


def iter_geng_family_graphs(family: FamilySpec, seed: int) -> Iterator[nx.Graph]:
    density_mode = family.name.replace("geng_", "")
    rng = random.Random(seed)

    counts_by_n = {}
    choices = [rng.randint(family.n_min, family.n_max) for _ in range(family.num_graphs)]
    for n in choices:
        counts_by_n[n] = counts_by_n.get(n, 0) + 1

    emitted = 0
    for n in sorted(counts_by_n.keys()):
        need = counts_by_n[n]
        graphs = sample_geng_graphs_for_n(
            n=n,
            need=need,
            density_mode=density_mode,
            rng=rng,
            mod=family.params.get("mod", 2000),
            max_tries=family.params.get("max_tries", 250),
        )
        for g in graphs:
            yield g
            emitted += 1

    while emitted < family.num_graphs:
        n = rng.randint(family.n_min, family.n_max)
        p = min(0.2, max(0.05, 4.0 / n))
        yield _connect_components(nx.erdos_renyi_graph(n=n, p=p))
        emitted += 1


# =========================================================
# Labels
# =========================================================

def build_graph_label(class_id: int) -> torch.Tensor:
    return torch.tensor(class_id, dtype=torch.long)


def build_node_labels_for_planted_clique(n: int, clique_nodes: List[int]) -> torch.Tensor:
    y = torch.zeros(n, dtype=torch.long)
    for v in clique_nodes:
        y[v] = 1
    return y


def build_node_labels_for_cycle_membership(n: int, cycle_nodes: List[int]) -> torch.Tensor:
    y = torch.zeros(n, dtype=torch.long)
    for v in cycle_nodes:
        y[v] = 1
    return y


# =========================================================
# Data builders
# =========================================================

def build_data_from_graph(
    g: nx.Graph,
    family_name: str,
    class_label: int,
    task_type: str,
    feature_regime: str,
    seed: int,
    aux: Optional[Dict[str, Any]] = None,
) -> Data:
    aux = aux or {}
    n = g.number_of_nodes()
    x = make_node_features(g, feature_regime, seed=seed)

    if task_type == "graph":
        y = build_graph_label(class_label)

    elif task_type == "node":
        if family_name == "planted_clique":
            y = build_node_labels_for_planted_clique(n, aux.get("clique_nodes", []))
        elif family_name == "hole_rich":
            y = build_node_labels_for_cycle_membership(n, aux.get("cycle_nodes", []))
        elif family_name == "triangle_rich":
            tri = nx.triangles(g)
            y = torch.tensor([1 if tri[v] > 0 else 0 for v in range(n)], dtype=torch.long)
        else:
            y = torch.zeros(n, dtype=torch.long)

    elif task_type == "edge":
        if family_name == "planted_clique":
            clique_set = set(aux.get("clique_nodes", []))
            edge_labels = {}
            for u, v in g.edges():
                edge_labels[(u, v)] = 1 if (u in clique_set and v in clique_set) else 0
            y = edge_label_tensor_from_dict(g, edge_labels)
        elif family_name == "hole_rich":
            y = edge_label_tensor_from_dict(g, aux.get("cycle_edge_labels", {}))
        elif family_name == "triangle_rich":
            tri = nx.triangles(g)
            edge_labels = {}
            for u, v in g.edges():
                edge_labels[(u, v)] = 1 if (tri[u] > 0 and tri[v] > 0) else 0
            y = edge_label_tensor_from_dict(g, edge_labels)
        else:
            y = torch.zeros(g.number_of_edges(), dtype=torch.long)

    elif task_type == "pair":
        y = build_pair_labels_from_graph(g)

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    data = Data(
        x=x,
        edge_index=nx_to_edge_index(g),
        y=y,
        num_nodes=n,
    )
    data.family = family_name
    data.graph_label = class_label
    data.num_edges_undirected = g.number_of_edges()
    data.graph6 = graph6_from_nx(g)
    return data


def build_graph_record(g: Data, family_local_index: int, global_index: int) -> Dict[str, Any]:
    return {
        "global_index": global_index,
        "family": getattr(g, "family", None),
        "family_local_index": family_local_index,
        "graph_label": int(getattr(g, "graph_label", -1)),
        "num_nodes": int(g.num_nodes),
        "num_edges": int(getattr(g, "num_edges_undirected", g.edge_index.size(1) // 2)),
        "x_shape": list(g.x.shape) if getattr(g, "x", None) is not None else None,
        "y_shape": list(g.y.shape) if getattr(g, "y", None) is not None and hasattr(g.y, "shape") else [],
        "graph6": getattr(g, "graph6", None),
    }


def iter_data_from_family(
    family: FamilySpec,
    task_type: str,
    feature_regime: str,
    graph_seed: int,
) -> Iterator[Data]:
    if family.name in {"geng_sparse", "geng_medium", "geng_dense", "geng_mixed"}:
        for i, g in enumerate(iter_geng_family_graphs(family, seed=graph_seed)):
            yield build_data_from_graph(
                g=g,
                family_name=family.name,
                class_label=family.label,
                task_type=task_type,
                feature_regime=feature_regime,
                seed=graph_seed + i,
            )
        return

    rng = random.Random(graph_seed)

    for i in range(family.num_graphs):
        n = rng.randint(family.n_min, family.n_max)

        if family.name == "er":
            g = build_er_graph(n, family.params)
            aux = {}

        elif family.name == "triangle_rich":
            g = build_triangle_rich_graph(n, family.params)
            aux = {}

        elif family.name == "planted_clique":
            g, clique_nodes = build_planted_clique_graph(n, family.params)
            aux = {"clique_nodes": clique_nodes}

        elif family.name == "hole_rich":
            g, cycle_nodes, cycle_edge_labels = build_hole_rich_graph(n, family.params)
            aux = {
                "cycle_nodes": cycle_nodes,
                "cycle_edge_labels": cycle_edge_labels,
            }

        else:
            raise ValueError(f"Unsupported family: {family.name}")

        yield build_data_from_graph(
            g=g,
            family_name=family.name,
            class_label=family.label,
            task_type=task_type,
            feature_regime=feature_regime,
            seed=graph_seed + i,
            aux=aux,
        )


# =========================================================
# Splits
# =========================================================

def make_iid_split_from_records(num_graphs: int, cfg: SplitConfig, seed: int) -> Tuple[List[int], List[int], List[int]]:
    idxs = list(range(num_graphs))
    random.Random(seed).shuffle(idxs)

    n_train = int(cfg.train_ratio * num_graphs)
    n_val = int(cfg.val_ratio * num_graphs)

    train_ids = idxs[:n_train]
    val_ids = idxs[n_train:n_train + n_val]
    test_ids = idxs[n_train + n_val:]
    return train_ids, val_ids, test_ids


def make_size_extrapolation_split_from_records(records: List[Dict[str, Any]], cfg: SplitConfig) -> Tuple[List[int], List[int], List[int]]:
    assert cfg.train_max_nodes is not None
    assert cfg.val_max_nodes is not None

    train_ids, val_ids, test_ids = [], [], []
    for i, rec in enumerate(records):
        n = int(rec["num_nodes"])
        if n <= cfg.train_max_nodes:
            train_ids.append(i)
        elif n <= cfg.val_max_nodes:
            val_ids.append(i)
        else:
            test_ids.append(i)
    return train_ids, val_ids, test_ids


def make_distribution_shift_split_from_records(records: List[Dict[str, Any]], cfg: SplitConfig) -> Tuple[List[int], List[int], List[int]]:
    assert cfg.train_families is not None
    assert cfg.test_families is not None

    train_set = set(cfg.train_families)
    val_set = set(cfg.val_families) if cfg.val_families is not None else set()
    test_set = set(cfg.test_families)

    train_ids, val_ids, test_ids = [], [], []
    for i, rec in enumerate(records):
        fam = rec["family"]
        if fam in test_set:
            test_ids.append(i)
        elif fam in val_set:
            val_ids.append(i)
        elif fam in train_set:
            train_ids.append(i)

    return train_ids, val_ids, test_ids


def make_split_from_records(records: List[Dict[str, Any]], cfg: SplitConfig, seed: int) -> Tuple[List[int], List[int], List[int]]:
    if cfg.scheme == "iid":
        return make_iid_split_from_records(len(records), cfg, seed)
    if cfg.scheme == "size_extrapolation":
        return make_size_extrapolation_split_from_records(records, cfg)
    if cfg.scheme == "distribution_shift":
        return make_distribution_shift_split_from_records(records, cfg)
    raise ValueError(f"Unknown split scheme: {cfg.scheme}")


# =========================================================
# Raw IO
# =========================================================

def append_graph_to_pickle(graph_path: str, graph: Data, fsync: bool = False):
    with open(graph_path, "ab") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        if fsync:
            os.fsync(f.fileno())


def load_pickled_graphs(graph_path: str) -> List[Data]:
    graphs = []
    with open(graph_path, "rb") as f:
        while True:
            try:
                graphs.append(pickle.load(f))
            except EOFError:
                break
    return graphs


# =========================================================
# Build + save in organized family folders
# =========================================================

def build_and_save_dataset(datasets_root: str, cfg: DatasetConfig, write_global_copy: bool = True):
    """
    Saves raw data to:

      datasets_root/<cfg.name>/raw/
        metadata.json
        splits.pt
        all/graphs.pkl                 (optional)
        <family_name>/graphs.pkl       (always)
    """
    set_seed(cfg.seed)

    dataset_root = osp.join(datasets_root, cfg.name)
    raw_dir = osp.join(dataset_root, "raw")
    ensure_dir(raw_dir)

    metadata_path = osp.join(raw_dir, "metadata.json")
    splits_path = osp.join(raw_dir, "splits.pt")

    all_graphs_path = osp.join(raw_dir, "all", "graphs.pkl")
    if write_global_copy:
        reset_file(all_graphs_path)

    family_graph_paths = {}
    for fam in cfg.families:
        family_dir = osp.join(raw_dir, fam.name)
        ensure_dir(family_dir)
        family_graph_path = osp.join(family_dir, "graphs.pkl")
        reset_file(family_graph_path)
        family_graph_paths[fam.name] = family_graph_path

    records: List[Dict[str, Any]] = []
    family_counts: Dict[str, int] = {}
    running_seed = cfg.seed * 1000 + 17
    global_index = 0

    for fam in cfg.families:
        family_counts[fam.name] = 0
        family_path = family_graph_paths[fam.name]

        for g in iter_data_from_family(
            family=fam,
            task_type=cfg.task_type,
            feature_regime=cfg.feature_regime,
            graph_seed=running_seed,
        ):
            append_graph_to_pickle(family_path, g)
            if write_global_copy:
                append_graph_to_pickle(all_graphs_path, g)

            rec = build_graph_record(
                g,
                family_local_index=family_counts[fam.name],
                global_index=global_index,
            )
            records.append(rec)

            family_counts[fam.name] += 1
            global_index += 1

        running_seed += fam.num_graphs + 97

    train_ids, val_ids, test_ids = make_split_from_records(records, cfg.split, cfg.seed)

    torch.save(
        {"train": train_ids, "val": val_ids, "test": test_ids},
        splits_path,
    )

    metadata = {
        "config": {
            "name": cfg.name,
            "task_type": cfg.task_type,
            "feature_regime": cfg.feature_regime,
            "seed": cfg.seed,
            "families": [asdict(f) for f in cfg.families],
            "split": asdict(cfg.split),
        },
        "layout": {
            "raw_dir": raw_dir,
            "global_graphs_path": all_graphs_path if write_global_copy else None,
            "family_graphs": {
                fam.name: family_graph_paths[fam.name] for fam in cfg.families
            },
        },
        "family_counts": family_counts,
        "graphs": records,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved raw dataset '{cfg.name}' to {raw_dir}")
    if write_global_copy:
        print(f"Global raw graphs file: {all_graphs_path}")
    for fam in cfg.families:
        print(f"Family {fam.name}: {family_counts[fam.name]} graphs -> {family_graph_paths[fam.name]}")
    print(f"Total graphs: {len(records)}")
    print(f"Split sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")


# =========================================================
# Example
# =========================================================

if __name__ == "__main__":
    NUM_PER_FAMILY = 10000

    cfg = DatasetConfig(
        name="cosc-structural-graphs",
        task_type="pair",
        feature_regime="ones",
        seed=7,
        families=[
            FamilySpec(
                name="triangle_rich",
                num_graphs=NUM_PER_FAMILY,
                n_min=22,
                n_max=30,
                label=1,
                params={
                    "k": 6,
                    "rewire_p": 0.03,
                },
            ),
            FamilySpec(
                name="planted_clique",
                num_graphs=NUM_PER_FAMILY,
                n_min=22,
                n_max=30,
                label=2,
                params={
                    "clique_size": 5,
                    "p": 0.08,
                },
            ),
            FamilySpec(
                name="hole_rich",
                num_graphs=NUM_PER_FAMILY,
                n_min=22,
                n_max=30,
                label=3,
                params={
                    "num_cycles": 1,
                    "min_cycle_len": 8,
                },
            ),
        ],
        split=SplitConfig(
            scheme="iid",
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        ),
    )

    ROOT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    datasets_root = os.path.join(ROOT_DIR, "datasets")

    build_and_save_dataset(
        datasets_root=datasets_root,
        cfg=cfg,
        write_global_copy=True,
    )