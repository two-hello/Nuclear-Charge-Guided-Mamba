"""
split_generator.py
Fixed float-sum comparison in setup_random_split()
"""

import json
import logging
import os
import math
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import index2mask, set_dataset_attr
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

# --------------------------------------------------------------------------- #
# High-level dispatcher
# --------------------------------------------------------------------------- #
def prepare_splits(dataset):
    """Ready train/val/test splits."""
    split_mode = cfg.dataset.split_mode
    if split_mode == 'standard':
        setup_standard_split(dataset)
    elif split_mode == 'random':
        setup_random_split(dataset)
    elif split_mode.startswith('cv-'):
        cv_type, k = split_mode.split('-')[1:]
        setup_cv_split(dataset, cv_type, int(k))
    elif split_mode == 'scaffold':
        setup_scaffold_split(dataset)
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

# --------------------------------------------------------------------------- #
# Scaffold split helpers
# --------------------------------------------------------------------------- #
def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

def generate_scaffolds(dataset):
    scaffolds = {}
    for ind, smiles in enumerate(dataset.smiles_data):
        scaffold = _generate_scaffold(smiles)
        if scaffold:
            scaffolds.setdefault(scaffold, []).append(ind)
    return [sorted(indices) for indices in sorted(scaffolds.values(), key=len, reverse=True)]

def setup_scaffold_split(dataset):
    valid_size, test_size = cfg.dataset.split[1], cfg.dataset.split[2]
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)
    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)

    train_inds, valid_inds, test_inds = [], [], []
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    set_dataset_splits(dataset, [train_inds, valid_inds, test_inds])

# --------------------------------------------------------------------------- #
# Standard split
# --------------------------------------------------------------------------- #
def setup_standard_split(dataset):
    split_index = cfg.dataset.split_index
    task_level = cfg.dataset.task

    if task_level == 'node':
        for split_name in ('train_mask', 'val_mask', 'test_mask'):
            mask = getattr(dataset.data, split_name, None)
            if mask is None:
                raise ValueError(f"Missing '{split_name}' for standard split")
            if mask.dim() == 2:
                if split_index >= mask.shape[1]:
                    raise IndexError(
                        f"Split index {split_index} out of range "
                        f"({mask.shape[1]} splits available)"
                    )
                set_dataset_attr(dataset, split_name, mask[:, split_index],
                                 len(mask[:, split_index]))
            else:
                if split_index != 0:
                    raise IndexError("This dataset has single standard split")

    elif task_level == 'graph':
        for split_name in ('train_graph_index', 'val_graph_index', 'test_graph_index'):
            if not hasattr(dataset.data, split_name):
                raise ValueError(f"Missing '{split_name}' for standard split")
        if split_index != 0:
            raise NotImplementedError("Multiple standard splits not supported for graph-level")

    elif task_level == 'link_pred':
        for split_name in ('train_edge_index', 'val_edge_index', 'test_edge_index'):
            if not hasattr(dataset.data, split_name):
                raise ValueError(f"Missing '{split_name}' for standard split")
        if split_index != 0:
            raise NotImplementedError("Multiple standard splits not supported for link-level")
    else:
        raise ValueError(f"Unsupported task level: {task_level}")

# --------------------------------------------------------------------------- #
# Random split  (FIXED)
# --------------------------------------------------------------------------- #
def setup_random_split(dataset):
    split_ratios = cfg.dataset.split
    if len(split_ratios) != 3:
        raise ValueError(
            f"Three split ratios expected for train/val/test, got "
            f"{len(split_ratios)}: {split_ratios}"
        )
    if not math.isclose(sum(split_ratios), 1.0, rel_tol=1e-5):
        raise ValueError(
            f"The train/val/test split ratios must sum up to 1, "
            f"input ratios sum up to {sum(split_ratios):.6f}: {split_ratios}"
        )

    train_index, val_test_index = next(
        ShuffleSplit(train_size=split_ratios[0], random_state=cfg.seed)
        .split(dataset.data.y, dataset.data.y)
    )
    val_index, test_index = next(
        ShuffleSplit(
            train_size=split_ratios[1] / (1 - split_ratios[0]),
            random_state=cfg.seed
        ).split(dataset.data.y[val_test_index], dataset.data.y[val_test_index])
    )
    val_index = val_test_index[val_index]
    test_index = val_test_index[test_index]
    set_dataset_splits(dataset, [train_index, val_index, test_index])

# --------------------------------------------------------------------------- #
# Cross-validation split
# --------------------------------------------------------------------------- #
def setup_cv_split(dataset, cv_type, k):
    split_index = cfg.dataset.split_index
    split_dir = cfg.dataset.split_dir
    os.makedirs(split_dir, exist_ok=True)

    save_file = os.path.join(
        split_dir,
        f"{cfg.dataset.format}_{dataset.name}_{cv_type}-{k}.json"
    )
    if not os.path.isfile(save_file):
        create_cv_splits(dataset, cv_type, k, save_file)

    with open(save_file) as f:
        cv = json.load(f)

    assert cv['dataset'] == dataset.name
    assert cv['n_samples'] == len(dataset)
    assert cv['n_splits'] == k
    assert split_index < k

    test_ids = cv[str(split_index)]
    val_ids = cv[str((split_index + 1) % k)]
    train_ids = []
    for i in range(k):
        if i != split_index and i != (split_index + 1) % k:
            train_ids.extend(cv[str(i)])
    set_dataset_splits(dataset, [train_ids, val_ids, test_ids])

def create_cv_splits(dataset, cv_type, k, file_name):
    n_samples = len(dataset)
    if cv_type == 'stratifiedkfold':
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=123)
        split_iter = kf.split(np.zeros(n_samples), dataset.data.y)
    elif cv_type == 'kfold':
        kf = KFold(n_splits=k, shuffle=True, random_state=123)
        split_iter = kf.split(np.zeros(n_samples))
    else:
        raise ValueError(f"Unexpected CV type: {cv_type}")

    splits = {
        'n_samples': n_samples,
        'n_splits': k,
        'cross_validator': str(kf),
        'dataset': dataset.name,
    }
    for fold, (_, ids) in enumerate(split_iter):
        splits[str(fold)] = ids.tolist()

    with open(file_name, 'w') as f:
        json.dump(splits, f)
    logging.info(f"[*] Saved CV splits by {kf} to {file_name}")

# --------------------------------------------------------------------------- #
# Utility: apply splits to dataset object
# --------------------------------------------------------------------------- #
def set_dataset_splits(dataset, splits):
    for i in range(len(splits) - 1):
        for j in range(i + 1, len(splits)):
            if set(splits[i]) & set(splits[j]):
                raise ValueError(
                    f"Splits must not intersect: "
                    f"split {i} & {j} share elements"
                )

    task_level = cfg.dataset.task
    if task_level == 'node':
        names = ['train_mask', 'val_mask', 'test_mask']
        for name, idx in zip(names, splits):
            mask = index2mask(idx, size=dataset.data.y.shape[0])
            set_dataset_attr(dataset, name, mask, len(mask))
    elif task_level == 'graph':
        names = ['train_graph_index', 'val_graph_index', 'test_graph_index']
        for name, idx in zip(names, splits):
            set_dataset_attr(dataset, name, idx, len(idx))
    else:
        raise ValueError(f"Unsupported task level: {task_level}")