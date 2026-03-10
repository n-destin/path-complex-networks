from __future__ import annotations

import os.path as osp
import torch

from lib.data.datasets import InMemoryComplexDataset
from lib.utils.graph_to_complex import convert_graph_dataset_with_paths
from lib.utils.log_utils import makedirs
from lib.datasets.graph_dataset_package.lib.utils.generators.io import load_generated_dataset


class StructuredGraphDataset(InMemoryComplexDataset):
    """Reads a generated graph dataset from raw/<name>/ and converts it to complexes.

    Expected files inside raw/<name>/:
      - graphs.pt
      - splits.pt
      - metadata.json
    """

    def __init__(
        self,
        root=None,
        name=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        max_dim=None,
        num_classes=None,
        train_ids=None,
        test_ids=None,
        val_ids=None,
        include_down_adj=False,
        init_method='sum',
        complex_type='path',
        n_jobs=2,
        **kwargs,
    ):
        self.name = name
        self.root = root
        self._n_jobs = n_jobs
        self._init_method = init_method
        self.num_node_labels = None
        self.metadata = None

        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            max_dim,
            num_classes,
            include_down_adj,
            init_method,
            complex_type,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

        _, saved_train_ids, saved_val_ids, saved_test_ids, metadata = load_generated_dataset(self.root_raw_parent, self.name)
        self.metadata = metadata

        self.train_ids = saved_train_ids if train_ids is None else train_ids
        self.val_ids = saved_val_ids if val_ids is None else val_ids
        self.test_ids = saved_test_ids if test_ids is None else test_ids

    @property
    def root_raw_parent(self):
        return self.root

    @property
    def processed_dir(self):
        directory = super().processed_dir
        suffix = '_down_adj' if self.include_down_adj else ''
        return directory + suffix

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw', self.name)

    @property
    def processed_file_names(self):
        return [f'{self.name}_complex_list.pt']

    @property
    def raw_file_names(self):
        return ['graphs.pt', 'splits.pt', 'metadata.json']

    def download(self):
        pass

    def process(self):
        print(f'Converting {self.name} into path complexes...')
        graphs, train_ids, val_ids, test_ids, metadata = load_generated_dataset(self.root, self.name)
        complexes, max_dim, _ = convert_graph_dataset_with_paths(
            graphs,
            max_k=self._max_dim,
            include_down_adj=self.include_down_adj,
            init_edges=True,
            init_high_order_paths=True,
            init_method=self._init_method,
            n_jobs=self._n_jobs,
        )

        if max_dim != self.max_dim:
            self.max_dim = max_dim
            makedirs(self.processed_dir)

        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.metadata = metadata

        path = self.processed_paths[0]
        torch.save(self.collate(complexes, self.max_dim), path)
