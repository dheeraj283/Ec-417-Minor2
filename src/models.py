import numpy as np
import torch
import torch.nn as nn
from scripts.get_data import get_16qam_constellation, apply_saleh_pa

class IdealGridDetector:
    def __init__(self, match_power=True):
        constellation = get_16qam_constellation()
        if match_power:
            pa_points = apply_saleh_pa(0.7 * constellation)
            scale = np.sqrt(np.mean(np.abs(pa_points) ** 2))
            constellation = constellation * scale
        self.points = np.column_stack((constellation.real, constellation.imag))

    def predict(self, X):
        diff = X[:, None, :] - self.points[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        return np.argmin(dist_sq, axis=1)

class KnownPACentroidDetector:
    def __init__(self):
        constellation = get_16qam_constellation()
        distorted = apply_saleh_pa(0.7 * constellation)
        self.centroids = np.column_stack((distorted.real, distorted.imag))

    def predict(self, X):
        diff = X[:, None, :] - self.centroids[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        return np.argmin(dist_sq, axis=1)

class DemodulatorMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_classes=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.forward(x_tensor)
            preds = torch.argmax(logits, dim=1).numpy()
        return preds
