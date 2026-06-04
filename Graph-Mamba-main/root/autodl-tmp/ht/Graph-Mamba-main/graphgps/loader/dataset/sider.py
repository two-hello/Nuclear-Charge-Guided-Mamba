import hashlib
import os.path as osp
import pickle
import shutil

import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm


class SiderDataset(InMemoryDataset):
    def __init__(self, root='datasets', smiles2graph=smiles2graph,
                 transform=None, pre_transform=None):
        """
        PyG dataset of drugs represented as their molecular graph (SMILES)
        with associated side effect targets from the Sider dataset.

        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'sider-dataset')

        self.url = None  # Since the dataset is assumed to be already downloaded
        self.version = None  # You can set a version hash if needed
        self.url_stratified_split = None  # Set if there's a split file URL
        self.md5sum_stratified_split = None  # Set if there's a split file MD5 hash

        # Check if the dataset exists in the specified folder
        data_path = osp.join(self.folder, '../data')
        if not osp.exists(data_path):
            raise FileNotFoundError(f"Sider dataset not found in {data_path}. "
                                    "Please ensure it is downloaded correctly.")

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['sider.csv']  # Adjust according to the actual file name(s) of the dataset

    @property
    def processed_file_names(self):
        return 'sider_data_processed.pt'

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        pass  # Since the dataset is assumed to be already downloaded

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]))

        # 假设sider数据集中SMILES列名为'smiles'，目标列名为'side_effect_target'
        smiles_list = data_df['smiles']
        target_names = ['side_effect_target']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][target_names]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(
                torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([y])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(self.root, 'sider_dataset_splits.pickle')  # 假设分割文件名为'sider_dataset_splits.pickle'
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict


if __name__ == '__main__':
    dataset = SiderDataset()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())