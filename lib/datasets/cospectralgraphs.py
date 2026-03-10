import torch 
import os 
import os.path as osp
import ast
import math

from lib.data.datasets import InMemoryComplexDataset
from lib.utils.graph_to_complex import convert_graph_dataset_with_paths

from lib.utils.log_utils import makedirs
from lib.utils.cospectralgraphs_utils import load_cospectral_graphs
from torch_geometric.data import Data

class COSCDataset(InMemoryComplexDataset):
    def __init__(self, root=None, name = None, transform=None, pre_transform=None, pre_filter=None, max_dim = None, num_classes = None, train_ids = None, test_ids =  None, val_ids = None, include_down_adj=False, init_method = 'sum', complex_type='path', n_jobs = 2, **kwargs):
        self.name = name 
        self.root = root
        self._n_jobs = n_jobs 
        self._init_method = init_method
        self.num_node_labels = None
        super(COSCDataset, self).__init__(root, transform, pre_transform, pre_filter, max_dim, num_classes, include_down_adj, init_method, complex_type)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        self.train_ids = list(range(self.len())) if train_ids is None else train_ids
        self.val_ids = list(range(self.len())) if val_ids is None else train_ids
        self.test_ids = list(range(self.len())) if test_ids is None else test_ids

    @property
    def processed_dir(self):
        directory = super(COSCDataset, self).processed_dir
        suffix = f"_down_adj" if self.include_down_adj else ""
        return directory + suffix
    
    @property
    def raw_dir(self):
        return osp.join(self.root, "raw")

    @property
    def processed_file_names(self):
        return  ['{}_complex_list.pt'.format(self.name)]
    
    @property
    def raw_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        print("Converting the {} dataset into path complexes... ".format(self.name))
        graphs, train_ids, val_ids, test_ids = load_cospectral_dataset(self.root)
        complexes, max_dim, _ = convert_graph_dataset_with_paths(
            graphs, 
            max_k=self._max_dim, 
            include_down_adj=self.include_down_adj,
            init_edges=True, 
            init_high_order_paths=True ,
            init_method=self._init_method, 
            n_jobs=self._n_jobs
        )
        if max_dim != self.max_dim:
            self.max_dim = max_dim
            makedirs(self.processed_dir)
        
        path = self.processed_paths[0]
        
        self.train_ids = train_ids 
        self.val_ids = val_ids 
        self.test_ids = test_ids

        print("before saving", len(self.train_ids), len(train_ids))
        
        torch.save(self.collate(complexes, self.max_dim), path)


def load_cospectral_dataset(root, name = "cosc-graphs"):
    raw_dir = osp.join(root, 'raw')


    # when loading graph6 graphs uncomment the code
    '''
    
        graphs = list()
        data = load_cospectral_graphs(os.path.join(raw_dir, "graphs.g6"))
        labels = []
        with open(raw_dir + "/graphs_labels.txt", "r") as file:
            for line in file.readlines():
                line = ast.literal_eval(line.strip())
                labels.append(line)
        for index, datum in enumerate(data):
            edge_index, num_nodes = datum
            num_pairs = math.comb(num_nodes, 2)
            label = torch.tensor(labels[index]).reshape(num_pairs, 3)
            assert label.shape == (num_pairs, 3), f"wrong label dimension {label.shape}"
            x = torch.ones(num_nodes, 1, dtype=torch.float32) # dummy node levels 
            graph = Data(x=x, edge_index=edge_index, y=label, edge_attr=None, num_nodes=num_nodes)  # set y to be something here.
            graphs.append(graph)
        
    '''

    

    train_last_index = int(len(graphs) * 0.8)
    validate_last_index = int(len(graphs) * 0.9)
    train_ids = list(range(train_last_index))
    val_ids = list(range(train_last_index, validate_last_index))
    test_ids = list(range(validate_last_index, len(graphs)))
    
    return graphs, train_ids, val_ids, test_ids