import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

from .featurize import smiles_to_graph

SMILES_COLUMN = "smiles"
VALUE_COLUMN = "value"
LOGIT_CLIP = 20.0
RESAMPLE_TRIES = 5


def graph_to_data(g):
    x = np.asarray(g["x"], dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    edge_index = np.asarray(g["edge_index"], dtype=np.int64)
    if edge_index.ndim == 1:
        edge_index = edge_index.reshape(2, -1)

    edge_attr = np.asarray(g["edge_attr"], dtype=np.float32)
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.reshape(-1, 1)

    gl = np.asarray(g["g"], dtype=np.float32)
    if gl.ndim == 1:
        gl = gl.reshape(1, -1)

    return Data(
        x=torch.from_numpy(x).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr).float(),
        g=torch.from_numpy(gl).float(),
    )


def build_dataset(smiles, values=None, workers=1):
    from tqdm.auto import tqdm

    values = values if values is not None else [None] * len(smiles)
    records = []
    skipped = []

    if workers > 1:
        import multiprocessing as mp

        with mp.Pool(workers) as pool:
            graphs = list(tqdm(pool.imap(smiles_to_graph, smiles, chunksize=64),
                               total=len(smiles)))
    else:
        graphs = [smiles_to_graph(s) for s in tqdm(smiles)]

    for smi, val, g in zip(smiles, values, graphs):
        if g is None:
            skipped.append(smi)
            continue
        record = dict(g)
        record["smiles"] = smi
        if val is not None:
            record["value"] = float(val)
        records.append(record)

    return records, skipped


def save_dataset(records, path):
    np.savez_compressed(path, data=np.array(records, dtype=object))


class MolPool:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        records = np.load(path, allow_pickle=True)["data"]

        self.graphs = []
        self.smiles = []
        values = []
        for r in records:
            if not isinstance(r, dict):
                continue
            self.graphs.append(graph_to_data(r))
            self.smiles.append(r.get("smiles", ""))
            values.append(float(r["value"]) if "value" in r else np.nan)

        if not self.graphs:
            raise RuntimeError(f"no usable molecules in {path}")

        self.values = np.asarray(values, dtype=np.float64)
        self.sorted_idx = np.argsort(self.values)
        self.sorted_values = self.values[self.sorted_idx]
        self.labeled = bool(np.isfinite(self.values).all())

    def __len__(self):
        return len(self.graphs)

    def graph(self, i):
        return self.graphs[i]

    def value(self, i):
        return float(self.values[i])


class PairDataset(Dataset):
    def __init__(self, pool, n_pairs, temperature, hard_ratio, min_delta, window):
        self.pool = pool
        self.n_pairs = int(n_pairs)
        self.temperature = float(temperature)
        self.hard_ratio = float(hard_ratio)
        self.min_delta = float(min_delta)
        self.outer = float(min_delta) + float(window)

    def __len__(self):
        return self.n_pairs

    def _sample_near(self, value):
        n = len(self.pool)
        sv = self.pool.sorted_values
        lo_out = int(np.searchsorted(sv, value - self.outer, side="left"))
        hi_out = int(np.searchsorted(sv, value + self.outer, side="right"))

        if self.min_delta > 0.0:
            lo_in = int(np.searchsorted(sv, value - self.min_delta, side="left"))
            hi_in = int(np.searchsorted(sv, value + self.min_delta, side="right"))
            left = max(0, lo_in - lo_out)
            right = max(0, hi_out - hi_in)
            if left + right == 0:
                return np.random.randint(0, n)
            r = np.random.randint(0, left + right)
            pos = (lo_out + r) if r < left else (hi_in + (r - left))
        else:
            if hi_out <= lo_out:
                return np.random.randint(0, n)
            pos = np.random.randint(lo_out, hi_out)

        return int(self.pool.sorted_idx[pos])

    def __getitem__(self, _):
        n = len(self.pool)
        ia = np.random.randint(0, n)
        va = self.pool.value(ia)

        hard = self.hard_ratio > 0.0 and np.random.rand() < self.hard_ratio
        ib = self._sample_near(va) if hard else np.random.randint(0, n)

        for _ in range(RESAMPLE_TRIES):
            if ib == ia:
                ib = self._sample_near(va) if hard else np.random.randint(0, n)
                continue
            if not hard and self.min_delta > 0.0 and abs(va - self.pool.value(ib)) < self.min_delta:
                ib = np.random.randint(0, n)
                continue
            break

        vb = self.pool.value(ib)

        z = float(np.clip((va - vb) / max(self.temperature, 1e-6), -LOGIT_CLIP, LOGIT_CLIP))
        y = 1.0 / (1.0 + np.exp(-z))

        return {"a": self.pool.graph(ia), "b": self.pool.graph(ib), "y": y}


def collate_pairs(batch):
    return {
        "a": Batch.from_data_list([b["a"] for b in batch]),
        "b": Batch.from_data_list([b["b"] for b in batch]),
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.float),
    }


def make_loader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
    )


def read_table(path):
    df = pd.read_csv(path)
    if SMILES_COLUMN not in df.columns:
        raise KeyError(
            f"{path} needs a '{SMILES_COLUMN}' column, found {list(df.columns)}")
    values = None
    if VALUE_COLUMN in df.columns:
        df = df.dropna(subset=[VALUE_COLUMN])
        values = df[VALUE_COLUMN].astype(float).tolist()
    return df[SMILES_COLUMN].astype(str).tolist(), values
