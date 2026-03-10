from __future__ import annotations

import json
import os
import os.path as osp
from typing import Dict, List, Tuple
import pickle


import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_generated_dataset(raw_root: str, dataset_name: str, graphs, train_ids, val_ids, test_ids, metadata: List[Dict], config: Dict):
    out_dir = osp.join(raw_root, dataset_name)
    ensure_dir(out_dir)
    torch.save(graphs, osp.join(out_dir, 'graphs.pt'))
    torch.save({'train': train_ids, 'val': val_ids, 'test': test_ids}, osp.join(out_dir, 'splits.pt'))
    with open(osp.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump({'config': config, 'graphs': metadata}, f, indent=2)


def load_generated_dataset(root: str, dataset_name: str):
    raw_root = osp.join(root, "raw")

    graphs = []
    for fam in ["hole_rich", "planted_clique", "triangle_rich"]:
        in_dir = osp.join(raw_root, fam)
        pkl_path = osp.join(in_dir, "graphs.pkl")
        with open(pkl_path, "rb") as f:
            while True:
                try:
                    graphs.append(pickle.load(f))
                except EOFError:
                    break

    splits = torch.load(osp.join(raw_root, "splits.pt"))
    with open(osp.join(raw_root, "metadata.json"), "r") as f:
        metadata = json.load(f)


    return graphs, splits["train"], splits["val"], splits["test"], metadata