from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


def _sample_n(rng: np.random.Generator, n_min: int, n_max: int) -> int:
    return int(rng.integers(n_min, n_max + 1))


def make_er_graph(rng: np.random.Generator, n_min: int, n_max: int, params: Dict) -> nx.Graph:
    n = _sample_n(rng, n_min, n_max)
    p = float(params.get('p', 0.12))
    g = nx.erdos_renyi_graph(n=n, p=p, seed=int(rng.integers(1_000_000_000)))
    if not nx.is_connected(g) and g.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
    return nx.convert_node_labels_to_integers(g)


def make_triangle_rich_graph(rng: np.random.Generator, n_min: int, n_max: int, params: Dict) -> nx.Graph:
    n = _sample_n(rng, n_min, n_max)
    k = int(params.get('k', min(6, max(2, n // 6))))
    if k % 2 == 1:
        k += 1
    rewire_p = float(params.get('rewire_p', 0.05))
    g = nx.watts_strogatz_graph(
        n=n,
        k=min(max(2, k), n - (n % 2 == 0 and 1 or 0)),
        p=rewire_p,
        seed=int(rng.integers(1_000_000_000)),
    )
    return nx.convert_node_labels_to_integers(g)


def make_planted_clique_graph(rng: np.random.Generator, n_min: int, n_max: int, params: Dict) -> Tuple[nx.Graph, List[int]]:
    n = _sample_n(rng, n_min, n_max)
    p = float(params.get('base_p', 0.08))
    clique_size = int(params.get('clique_size', max(4, n // 5)))
    clique_size = min(clique_size, n)

    g = nx.erdos_renyi_graph(n=n, p=p, seed=int(rng.integers(1_000_000_000)))
    clique_nodes = rng.choice(n, size=clique_size, replace=False).tolist()
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            g.add_edge(clique_nodes[i], clique_nodes[j])
    return nx.convert_node_labels_to_integers(g), clique_nodes


def make_hole_rich_graph(rng: np.random.Generator, n_min: int, n_max: int, params: Dict) -> Tuple[nx.Graph, List[int]]:
    """Construct a graph with one or more long cycles and few chords.

    Returns the graph and the nodes that lie on the planted cycle backbone.
    """
    n = _sample_n(rng, n_min, n_max)
    num_cycles = int(params.get('num_cycles', 1))
    min_cycle_len = int(params.get('min_cycle_len', max(4, n // 4)))
    tree_attach_prob = float(params.get('tree_attach_prob', 0.35))

    g = nx.Graph()
    g.add_nodes_from(range(n))
    available = list(range(n))
    rng.shuffle(available)

    cycle_nodes_all: List[int] = []
    cursor = 0
    for _ in range(num_cycles):
        remaining = n - cursor
        if remaining < min_cycle_len:
            break
        cycle_len = min(min_cycle_len, remaining)
        cyc = available[cursor: cursor + cycle_len]
        cursor += cycle_len
        cycle_nodes_all.extend(cyc)
        for i in range(cycle_len):
            g.add_edge(cyc[i], cyc[(i + 1) % cycle_len])

    leftovers = available[cursor:]
    attach_candidates = cycle_nodes_all[:] if cycle_nodes_all else available[:1]
    for v in leftovers:
        parent = int(rng.choice(attach_candidates))
        g.add_edge(v, parent)
        if rng.random() < tree_attach_prob:
            attach_candidates.append(v)

    if g.number_of_edges() == 0 and n >= 2:
        g.add_edge(0, 1)

    return nx.convert_node_labels_to_integers(g), cycle_nodes_all


FAMILY_BUILDERS = {
    'er': make_er_graph,
    'triangle_rich': make_triangle_rich_graph,
    'planted_clique': make_planted_clique_graph,
    'hole_rich': make_hole_rich_graph,
}
