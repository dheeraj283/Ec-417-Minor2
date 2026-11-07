import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from scripts.get_data import generate_samples, get_16qam_constellation, apply_saleh_pa

def add_polar_features(X):
    i = X[:, 0]
    q = X[:, 1]
    r = np.sqrt(i**2 + q**2)
    theta = np.arctan2(q, i)
    return np.column_stack((i, q, r, theta)).astype(np.float32)

def load_data(split="train", use_polar=False):
    path = os.path.join("data", f"{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run scripts/get_data.py first.")
    
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    
    if use_polar:
        X = add_polar_features(X)
        
    return X, y

def get_dataloader(X, y, batch_size=128, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
