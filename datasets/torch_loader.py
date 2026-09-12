#!/usr/bin/env python3
"""MicroTomo Dataset Torch Loader — HDF5 -> PyTorch DataLoader.

Reads batch-generated HDF5 datasets and provides a PyTorch Dataset
for training reconstruction networks (DAS baseline, U-Net, etc.).

Usage (in training code):
    from datasets.torch_loader import MicroTomoDataset
    ds = MicroTomoDataset("datasets/h5/train.h5")
    loader = DataLoader(ds, batch_size=16, shuffle=True)

HDF5 layout produced by scripts/batch_generate.py:
    /samples/NN/x            (F, T) complex64  — Tx ramp signal (F freqs x T times)
    /samples/NN/x_hat        (F, T) complex64  — background-subtracted
    /samples/NN/y            (F, T) complex64  — scattered signal
    /samples/NN/geometry     (H, W, D) float32 — phantom voxel mask
    /samples/NN/eps          (H, W, D) float32 — permittivity volume
    /samples/NN/meta         json str          — parameters of this sample
"""
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MicroTomoDataset(Dataset):
    """PyTorch dataset over MicroTomo HDF5 batches."""

    def __init__(self, h5_path, mode="scattered", normalize=True, transform=None):
        """Initialize.

        Args:
            h5_path: Path to HDF5 file produced by batch_generate.py
            mode: 'full' | 'scattered' | 'background' — which signal to return as x
            normalize: If True, scale to [-1, 1] using global stats at load
            transform: Optional callable applied to (x, y) pair
        """
        self.h5_path = h5_path
        self.mode = mode
        self.normalize = normalize
        self.transform = transform

        with h5py.File(h5_path, "r") as f:
            self.n_samples = len(f["samples"])
            self.n_freq = f["samples/0/x"].shape[0]
            self.n_time = f["samples/0/x"].shape[1]
            shape = f["samples/0/eps"].shape
            self.grid_shape = tuple(shape)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            grp = f[f"samples/{idx}"]
            x = grp["x"][:]
            y = grp["y"][:]
            eps = grp["eps"][:]
            meta = json.loads(grp["meta"][()])

        # Choose signal mode
        if self.mode == "full":
            sig = x
        elif self.mode == "scattered":
            sig = y
        elif self.mode == "background":
            sig = grp["x_hat"][:]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Convert to torch tensors (complex -> 2 real channels)
        x_real = torch.from_numpy(np.real(sig)).float()
        x_imag = torch.from_numpy(np.imag(sig)).float()
        x_torch = torch.stack([x_real, x_imag], dim=0)  # (2, F, T)

        y_real = torch.from_numpy(np.real(y)).float()
        y_imag = torch.from_numpy(np.imag(y)).float()
        y_torch = torch.stack([y_real, y_imag], dim=0)

        eps_torch = torch.from_numpy(eps).float().unsqueeze(0)  # (1, H, W, D)

        if self.normalize:
            # Min-max per sample
            x_torch = (x_torch - x_torch.min()) / (x_torch.max() - x_torch.min() + 1e-8)
            y_torch = (y_torch - y_torch.min()) / (y_torch.max() - y_torch.min() + 1e-8)

        if self.transform:
            x_torch, y_torch, eps_torch = self.transform(x_torch, y_torch, eps_torch)

        return {
            "x": x_torch,
            "y": y_torch,
            "eps": eps_torch,
            "meta": meta,
        }


class MicroTomoCollate:
    """Collate function that works with the dict-returning dataset."""

    @staticmethod
    def collate(batch):
        x = torch.stack([b["x"] for b in batch])
        y = torch.stack([b["y"] for b in batch])
        eps = torch.stack([b["eps"] for b in batch])
        meta = [b["meta"] for b in batch]
        return x, y, eps, meta


def quick_demo():
    """Sanity check: print dataset shape + one sample summary."""
    import sys
    from torch.utils.data import DataLoader

    path = sys.argv[1] if len(sys.argv) > 1 else "datasets/h5/train.h5"
    ds = MicroTomoDataset(path)
    print(f"Samples: {len(ds)}, freq bins: {ds.n_freq}, time steps: {ds.n_time}")
    print(f"Grid: {ds.grid_shape}")
    loader = DataLoader(ds, batch_size=2, collate_fn=MicroTomoCollate.collate)
    x, y, eps, meta = next(iter(loader))
    print(f"Batch shapes: x={tuple(x.shape)} y={tuple(y.shape)} eps={tuple(eps.shape)}")
    print(f"Meta keys: {list(meta[0].keys()) if isinstance(meta[0], dict) else meta}")


if __name__ == "__main__":
    quick_demo()