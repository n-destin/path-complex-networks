from torch_geometric.datasets import LRGBDataset
from lib.utils.graph_to_complex import convert_graph_dataset_with_gudhi, convert_graph_dataset_with_rings, convert_graph_dataset_with_paths
from lib.data.datasets import InMemoryComplexDataset
import os 
import os.path as osp
import torch

class LRGBData(InMemoryComplexDataset):
    
    def __init__(self, root = None, name : str = None, transform=None, pre_transform=None, 
                 pre_filter=None, max_dim = None, num_classes = None, 
                 include_down_adj=False, init_method = 'sum', 
                 complex_type='path',):
        super().__init__(root, transform, pre_transform, 
                         pre_filter, max_dim, num_classes, 
                         include_down_adj, init_method, 
                         complex_type)
        self.processed_path = None
        self.root = root
        self.name = name

        self.data, self.slices, index = self.load_dataset()
        self.train_ids = index[0]
        self.val_ids = index[1]
        self.test_ids = index[2]

        

    @property
    def raw_dir(self, ):
        return osp.join(self.root, "raw")
    
    @property
    def processed_dir(self, ): # the processed_paths takes this and appends the filenames to the paths
        return osp.join(self.root, "processed")

    # def download(self, ):
    #     for split in self.splits:
    #         dataset = LRGBDataset(self.root, self.name)
    #         self.processed_path = dataset.processed_dir



    def process(self, data):
        train_data = LRGBDataset(self.raw_dir, self.name, "train")
        val_data = LRGBDataset(self.raw_dir, self.name, "val")
        test_data = LRGBData(self.raw_dir, self.name, "test")

        data_list = []
        idx = [] # assigning indices to the complexes 
        start = 0
        
        print("Converting training dataset to {} complex".format(self._complex_type))
        train_complexes = self.convert_to_complex(train_data)
        data_list += train_complexes
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)

        print("Converting Validation dataset to {} complex".format(self._complex_type))
        validation_complexes = self.convert_to_complex(val_data)
        data_list += validation_complexes
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)

        print("Converting testing dataset to {} complex".format(self._complex_type))
        test_complexes = self.convert_to_complex(test_data)
        data_list += test_complexes
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)

        path = self.processed_paths[0] # 
        print(f'Saving processed dataset in {path}....')
        torch.save(self.collate(data_list, self.max_dim), path) # saves data and slices
        
        path = self.processed_paths[1]
        print(f'Saving idx in {path}....')
        torch.save(idx, path)

    def load_dataset(self): # from where it was saved
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        return data, slices, idx


    def convert_to_complex(self, data):
        if self._complex_type == 'simplicial':
            complexes, _, _, = convert_graph_dataset_with_gudhi(        # this function doesn't encode edge_attr, so must set use_edge_features False
                data,
                expansion_dim = self._max_dim,
                include_down_adj = self.include_down_adj,
                init_method = self._init_method
            )
        elif self._complex_type == 'cell':
            complexes, _, _ = convert_graph_dataset_with_rings(
                data,
                max_ring_size=self._max_ring_size,
                include_down_adj=self.include_down_adj,
                init_edges=self._use_edge_features,
                init_rings=False,
                n_jobs=self._n_jobs)
        elif self._complex_type == 'path':
            complexes, _, _ = convert_graph_dataset_with_paths(
                data,
                max_k = self._max_dim,
                include_down_adj=self.include_down_adj,
                init_edges=self._use_edge_features,
                init_high_order_paths=False,
                init_method = self._init_method,
                n_jobs=self._n_jobs)
        else:
            raise ValueError("Complex type not supported for this dataset")
        return complexes



##################################################
# LRGBData Graph #####################################
##################################################


def load_lrgb_dataset(root, name):
    raw_dir = osp.join(root, 'ZINC', 'raw')

    train_data = LRGBDataset(raw_dir, split='train') # this just loads the data into Data objects
    val_data = LRGBDataset(raw_dir, name=name, split='val')
    test_data = LRGBDataset(raw_dir, name=name, split='test')
    data = train_data + val_data + test_data

    idx = []
    start = 0
    idx.append(list(range(start, len(train_data))))
    start = len(train_data)
    idx.append(list(range(start, start + len(val_data))))
    start = len(train_data) + len(val_data)
    idx.append(list(range(start, start + len(test_data))))
    num_classes = 1 # change

    if name == "Peptides-func":
        num_classes = 10

    num_features = train_data[0].x.shape[1] # number of features of the nodes
    return data, idx[0], idx[1], idx[2], num_classes, num_features