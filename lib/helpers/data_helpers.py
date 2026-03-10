"""
Code is adapted from https://github.com/rusty1s/pytorch_geometric/blob/6442a6e287563b39dae9f5fcffc52cd780925f89/torch_geometric/data/dataloader.py

Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
Copyright (c) 2021 The CWN Project Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import os
import torch
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch
import collections.abc as container_abcs

from definitions import ROOT_DIR

from lib.data.cochain import Cochain, CochainBatch
from lib.data.complex import Complex, ComplexBatch
from lib.data.datasets import ComplexDataset
from lib.utils.random_seed import my_worker_init_fn
from lib.datasets.zinc import ZincDataset, load_zinc_graph_dataset
from lib.datasets.ogb import OGBDataset, load_ogb_graph_dataset
from lib.datasets.tu import TUDataset, load_tu_graph_dataset
from lib.datasets.sr import SRDataset, load_sr_graph_dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from lib.datasets.lrgb import LRGBData, load_lrgb_dataset
from lib.datasets.cospectralgraphs import COSCDataset, load_cospectral_dataset
from lib.datasets.graph_dataset_package.lib.data.datasets.structured_graph_dataset import StructuredGraphDataset, load_generated_dataset
int_classes = int
string_classes = str


class Collater(object):

    """Object that converts python lists of objects into the appropiate storage format.

    Args:
        follow_batch: Creates assignment batch vectors for each key in the list.
        max_dim: The maximum dimension of the cochains considered from the supplied list.
    """
    def __init__(self, follow_batch, max_dim=2):
        self.follow_batch = follow_batch
        self.max_dim = max_dim

    def collate(self, batch):
        """Converts a data list in the right storage format."""
        elem = batch[0]
        if isinstance(elem, Cochain):
            return CochainBatch.from_cochain_list(batch, self.follow_batch)
        elif isinstance(elem, Complex):
            return ComplexBatch.from_complex_list(batch, self.follow_batch, max_dim=self.max_dim)
        elif isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)
    

class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges cochain complexes into to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
        max_dim (int): The maximum dimension of the chains to be used in the batch.
            (default: 2)
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 max_dim=2, **kwargs):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for Pytorch Lightning...
        self.follow_batch = follow_batch

        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=Collater(follow_batch, max_dim),
                             worker_init_fn=my_worker_init_fn, pin_memory=False,                # OGB must set pin_memory = False
                             **kwargs)
        

# this is for loading teh data for complex models
def load_dataset(name, root=os.path.join(ROOT_DIR, 'datasets'), **kwargs) -> ComplexDataset:
    """Returns a ComplexDataset with the specified name and initialised with the given params."""
    if name == 'ZINC':
        dataset = ZincDataset(os.path.join(root, name), **kwargs)
    elif name == 'ZINC-FULL':
        dataset = ZincDataset(os.path.join(root, name), subset=False, **kwargs)
    # elif name in ['MOLHIV', 'MOLPCBA', 'MOLTOX21', 'MOLTOXCAST', 'MOLMUV',
    #               'MOLBACE', 'MOLBBBP', 'MOLCLINTOX', 'MOLSIDER', 'MOLESOL',
    #               'MOLFREESOLV', 'MOLLIPO']:
    #     official_name = 'ogbg-'+name.lower()
    #     type = 'graph-prediction'
    #     dataset = OGBDataset(os.path.join(root, name), type, official_name, simple=kwargs['simple_features'], **kwargs)
    elif name in ['PRODUCTS', 'PROTEINS', 'ARXIV', 'PAPERS100M', "MAG"]:
        official_name = 'ogbn-'+name.lower()
        dataset = OGBDataset(os.path.join(root, name), "node", official_name, simple=kwargs['simple_features'], **kwargs)
    elif name in ['PPA', 'COLLAB', 'DDI', 'CITATION2', 'WIKIKG2', "BIOKG", "VESSEL"]:
        official_name = 'ogbl-'+name.lower()
        dataset = OGBDataset(os.path.join(root, name), "link", official_name, simple=kwargs['simple_features'], **kwargs)
    elif name in ['IMDBBINARY', 'IMDBMULTI']:
        dataset = TUDataset(os.path.join(root, name), name, degree_as_tag=True, **kwargs)
    elif name in ['REDDITBINARY', 'REDDITMULTI5K', 'PROTEINS', 'NCI1', 'NCI109', 'PTC', 'MUTAG']:
        dataset = TUDataset(os.path.join(root, name), name, degree_as_tag=False, **kwargs)
    elif name.startswith('sr'):
        dataset = SRDataset(os.path.join(root, 'SR-GRAPHS'), name, **kwargs)
    elif name in  ['pascalvoc-sp', 'coco-sp', 'pcqm-contact', 'peptides-func', 'peptides-struct' ]:
        dataset = LRGBData(root = os.path.join(root, name), name = name)
    elif name.startswith("cosc"):
        dataset = StructuredGraphDataset(root = os.path.join(root, name), name = name, **kwargs)
    else:
        raise NotImplementedError(name)
    return dataset


# this for loading the data for normal graph models
def load_graph_dataset(name, root=os.path.join(ROOT_DIR, 'datasets'), **kwargs):
    """
    Returns a graph dataset with the specified name and initialised with the given params.
    Input: name
    Output: graph_list, train_ids, val_ids, test_ids, number of labels
    """
    num_node_labels = None
    if name == 'ZINC': # this data is not not converted into complexe. 
        graph_list, train_ids, val_ids, test_ids, num_classes, num_features = load_zinc_graph_dataset(root=root)
    elif name == 'ZINC-FULL':
        graph_list, train_ids, val_ids, test_ids, num_classes, num_features = load_zinc_graph_dataset(root=root, subset=False) 
    elif name in ['PRODUCTS', 'PROTEINS', 'ARXIV', 'PAPERS100M', "MAG"]:
        official_name = 'ogbn-'+name.lower()
        if name == "PAPERS100M": official_name = official_name[:-1] + "M"
        graph_list, train_ids, val_ids, test_ids, num_classes, num_features = load_ogb_graph_dataset(os.path.join(root, name), "node", official_name)
    elif name in ['PPA', 'COLLAB', 'DDI', 'CITATION2', 'WIKIKG2', "BIOKG", "VESSEL"]:
        graph_list, train_ids, val_ids, test_ids, num_classes, num_features = load_ogb_graph_dataset(os.path.join(root, name), "link", 'ogbl-'+name.lower())
    elif name in ['IMDBBINARY', 'IMDBMULTI']:
        graph_list, train_ids, val_ids, test_ids, num_classes, num_features, num_node_labels = load_tu_graph_dataset(name, root=root, 
                                                                                                                     degree_as_tag=True,  
                                                                                                                     fold=kwargs['fold'], seed=kwargs['seed'])
    elif name in ['REDDITBINARY', 'REDDITMULTI5K', 'PROTEINS', 'NCI1', 'NCI109', 'PTC', 'MUTAG']:
        graph_list, train_ids, val_ids, test_ids, num_classes, num_features, num_node_labels = load_tu_graph_dataset(name, root=root, 
                                                                                                                     degree_as_tag=False,                                                                                     fold=kwargs['fold'], seed=kwargs['seed'])
    elif name.startswith('sr'):
        graph_list, train_ids, val_ids, test_ids = load_sr_graph_dataset(name, root=os.path.join(root, 'SR-GRAPHS'))
        num_classes = 32
        num_features = 1
    elif name in  ['pascalvoc-sp', 'coco-sp', 'pcqm-contact', 'peptides-func', 'peptides-struct' ]:
        graph_list, train_ids, val_ids, test_ids, num_classes, num_features = load_lrgb_dataset(root = os.path.join(root, name), name = name)
    elif name.startswith("cosc"):
        graph_list, train_ids, val_ids, test_ids, _ = load_generated_dataset(root=os.path.join(root, name), dataset_name = name)
        num_classes = 3
        num_features = 1
    else:
        raise NotImplementedError
    
    data = (graph_list, train_ids, val_ids, test_ids, num_classes, num_features, num_node_labels)
    return data

def get_dataloaders(args, fold = None): # this are the graph s

     # Load Datasets
     # for normal graphs 
    if (not args.model.endswith('cin')):          # G IN This loads normal graphs
        graph_list, train_idx, val_idx, test_idx, num_classes, num_features, num_node_labels = load_graph_dataset(args.dataset, seed=args.seed, fold=fold, max_ring_size=args.max_ring_size)
        
        # Instantiate data loaders
        if args.task != "node" and args.task != "pair" and args.dataset != "cosc-structural-graphs": 
            train_graphs = [graph_list[i] for i in train_idx]
            val_graphs = [graph_list[i] for i in val_idx]
            if test_idx is not None:
                test_graphs = [graph_list[i] for i in test_idx]
            batch_size = args.batch_size
        else:
            train_graphs = [graph_list[0]]
            val_graphs = [graph_list[0]]
            if test_idx is not None:
                test_graphs = [graph_list[0]]
            batch_size = 1
        train_loader = PyGDataLoader(train_graphs, batch_size=batch_size,
                            shuffle=True, num_workers=args.num_workers)
        
        valid_loader = PyGDataLoader(val_graphs, batch_size=batch_size,
                            shuffle=False, num_workers=args.num_workers)
        
        if test_idx is not None:
            test_loader = PyGDataLoader(test_graphs, batch_size=batch_size,
                                shuffle=False, num_workers=args.num_workers)
        else:
            test_loader = None
        
        # Get dataloaders
        res = (train_loader, valid_loader, test_loader, train_idx, val_idx, test_idx, num_classes, num_features)
    else:                                       # High-order GNNs. For Graph Complex datasets
        _, train_ids, val_ids, test_ids, num_classes, _, num_node_labels = load_graph_dataset(args.dataset, seed=args.seed, fold=fold, max_ring_size=args.max_ring_size)
        dataset = load_dataset(args.dataset, 
                               complex_type = args.complex_type, 
                               max_dim=args.max_dim,
                               seed=args.seed, 
                               fold=fold, 
                               init_method=args.init_method, 
                               flow_points=args.flow_points, 
                               flow_classes=args.flow_classes,
                               num_classes=num_classes,
                               num_node_labels=num_node_labels,
                               max_ring_size=args.max_ring_size,
                               use_edge_features=args.use_edge_features,
                               simple_features=args.simple_features,
                               include_down_adj=args.include_down_adj,
                               n_jobs=args.preproc_jobs,
                               train_orient=args.train_orient, 
                               test_orient=args.test_orient,
                               train_ids=train_ids, 
                               test_ids = test_ids, 
                               val_ids = val_ids
                               )
            
        # Get dataset split
        split_idx = dataset.get_idx_split()
        
        # TODO: dataset.get_split(split) might not work for LRGBDataset

        # Instantiate data loaders
        train_loader = DataLoader(dataset.get_split('train'), batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, max_dim=dataset.max_dim)
        valid_loader = DataLoader(dataset.get_split('valid'), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)
        test_split = split_idx.get("test", None)
        test_loader = None
        if test_split is not None:
            test_loader = DataLoader(dataset.get_split('test'), batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)
        # Get dataloaders
        res = (train_loader, valid_loader, test_loader, dataset)
    return res