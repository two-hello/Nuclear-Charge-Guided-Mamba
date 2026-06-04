import os
import re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from numpy.random import default_rng

# -------------------- OGB PCQM4Mv2 --------------------
from ogb.lsc import PCQM4Mv2Dataset

# -------------------- mamba --------------------
from mamba_ssm import Mamba


# ====================================================
# 1. SMILES Tokenizer（安全版）
# ====================================================
class SmilesTokenizer:
    def __init__(self):
        self.regex = re.compile(
            r"Cl|Br|Si|Se|Na|Mg|Al|Ca|Fe|Zn|Cu|Mn|Au|Ag|Hg|Pt|Ni|"
            r"B|C|N|O|S|P|F|I|H|\[.*?\]|\(|\)|\.|=|#|-|\+|\\|@|\/|:|\d+"
        )
        self.special_tokens = ["<PAD>", "<UNK>"]
        self.atoms = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se"]
        self.bonds = ["-", "=", "#", "/", "\\"]
        self.others = ["(", ")", ".", "@", "+"]

        self.tokens = self.special_tokens + self.atoms + self.bonds + self.others
        self.token2idx = {t: i for i, t in enumerate(self.tokens)}
        self.pad_idx = self.token2idx["<PAD>"]
        self.unk_idx = self.token2idx["<UNK>"]
        self.vocab_size = len(self.tokens)

    def encode(self, smiles):
        if not isinstance(smiles, str):
            return []
        return [self.token2idx.get(t, self.unk_idx)
                for t in self.regex.findall(smiles)]


# ====================================================
# 2. PCQM4Mv2 子集划分（版本无关）
# ====================================================
def get_pcqm_subset_indices(dataset, subset_ratio=0.1, seed=42):
    split_idx = dataset.get_idx_split()

    train_full = split_idx["train"]
    valid_full = split_idx["valid"]

    if isinstance(train_full, torch.Tensor):
        train_full = train_full.numpy()
    if isinstance(valid_full, torch.Tensor):
        valid_full = valid_full.numpy()

    rng = default_rng(seed)
    perm = rng.permutation(train_full)

    n_sub = int(len(perm) * subset_ratio)
    sub_train = perm[:n_sub]

    print(f"[Subset] train={len(sub_train)}, valid={len(valid_full)}")
    return sub_train, valid_full


# ====================================================
# 3. SMILES-only Dataset（data.csv.gz，完全兜底）
# ====================================================
class PCQM4MSmilesDataset(Dataset):
    def __init__(self, dataset, indices, tokenizer, root, max_len=256):
        self.dataset = dataset
        self.indices = indices.tolist() if isinstance(indices, torch.Tensor) else indices
        self.tokenizer = tokenizer
        self.max_len = max_len

        data_path = os.path.join(root, "pcqm4m-v2", "raw", "data.csv.gz")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing data.csv.gz at {data_path}")

        df = pd.read_csv(data_path, compression="gzip")
        self.smiles = df.iloc[:, 0].fillna("").astype(str).tolist()

        assert len(self.smiles) == len(dataset), "SMILES 数量与 dataset 不一致"

        print("✔ Using data.csv.gz (fallback)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        smiles = self.smiles[idx]

        token_ids = self.tokenizer.encode(smiles)[: self.max_len]
        if len(token_ids) == 0:
            token_ids = [self.tokenizer.pad_idx]

        _, y = self.dataset[idx]

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(y, dtype=torch.float),
        )


# ====================================================
# 4. Collate
# ====================================================
def collate_fn(pad_idx):
    def _fn(batch):
        seqs, labels = zip(*batch)
        seqs = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
        mask = (seqs != pad_idx).long()
        labels = torch.stack(labels).unsqueeze(-1)
        return seqs, mask, labels
    return _fn


# ====================================================
# 5. Mamba Model
# ====================================================
class MambaEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = x * mask.unsqueeze(-1)          # padding 清零
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        length = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / length


class MambaForMolecularProperty(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = MambaEncoder(vocab_size)
        self.head = nn.Linear(256, 1)

    def forward(self, x, mask):
        return self.head(self.encoder(x, mask))


# ====================================================
# 6. Train / Eval
# ====================================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for x, mask, y in tqdm(loader, desc="Training"):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        pred = model(x, mask)
        loss = nn.functional.l1_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        pred = model(x, mask)
        total += torch.abs(pred - y).sum().item()
        n += x.size(0)
    return total / n


# ====================================================
# 7. Main
# ====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset_root = "/data/ht/Graph-Mamba-main/datasets"
    batch_size = 1024
    epochs = 500
    lr = 1e-5
    subset_ratio = 0.1

    print("Initializing PCQM4Mv2Dataset...")
    dataset = PCQM4Mv2Dataset(root=dataset_root)
    print("PCQM4Mv2Dataset ready.")

    train_idx, valid_idx = get_pcqm_subset_indices(dataset, subset_ratio)

    tokenizer = SmilesTokenizer()

    train_ds = PCQM4MSmilesDataset(dataset, train_idx, tokenizer, dataset_root)
    valid_ds = PCQM4MSmilesDataset(dataset, valid_idx, tokenizer, dataset_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # 调试期必须 0
        collate_fn=collate_fn(tokenizer.pad_idx),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn(tokenizer.pad_idx),
    )

    model = MambaForMolecularProperty(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_mae = train_epoch(model, train_loader, optimizer, device)
        val_mae = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch:03d} | Train MAE {train_mae:.4f} | Val MAE {val_mae:.4f}")

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), "best_model_pcqm.pt")

    print("Training finished. Best Val MAE:", best_val)