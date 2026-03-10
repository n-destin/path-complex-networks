from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def make_splits(graphs, metadata: List[Dict], split_cfg, seed: int = 0):
    if split_cfg.scheme == 'iid':
        return _iid_split(len(graphs), split_cfg, seed)
    if split_cfg.scheme == 'size_extrapolation':
        return _size_split(metadata, split_cfg)
    if split_cfg.scheme == 'distribution_shift':
        return _distribution_shift_split(metadata, split_cfg)
    raise ValueError(f'Unknown split scheme: {split_cfg.scheme}')


def _iid_split(n: int, split_cfg, seed: int = 0):
    ids = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n_train = int(split_cfg.train_ratio * n)
    n_val = int(split_cfg.val_ratio * n)

    train_ids = ids[:n_train].tolist()
    val_ids = ids[n_train:n_train + n_val].tolist()
    test_ids = ids[n_train + n_val:].tolist()
    return train_ids, val_ids, test_ids


def _size_split(metadata: List[Dict], split_cfg):
    if split_cfg.train_max_nodes is None or split_cfg.val_max_nodes is None:
        raise ValueError('size_extrapolation requires train_max_nodes and val_max_nodes')

    train_ids, val_ids, test_ids = [], [], []
    for idx, meta in enumerate(metadata):
        n = int(meta['num_nodes'])
        if n <= split_cfg.train_max_nodes:
            train_ids.append(idx)
        elif n <= split_cfg.val_max_nodes:
            val_ids.append(idx)
        else:
            test_ids.append(idx)
    return train_ids, val_ids, test_ids


def _distribution_shift_split(metadata: List[Dict], split_cfg):
    train_families = set(split_cfg.train_families or [])
    val_families = set(split_cfg.val_families or [])
    test_families = set(split_cfg.test_families or [])

    train_ids, val_ids, test_ids = [], [], []
    for idx, meta in enumerate(metadata):
        fam = meta['family']
        if fam in train_families:
            train_ids.append(idx)
        elif fam in val_families:
            val_ids.append(idx)
        elif fam in test_families:
            test_ids.append(idx)
    return train_ids, val_ids, test_ids
