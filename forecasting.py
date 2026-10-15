import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


LOOKBACK = 168
HORIZON = 24


def calendar_features(index):
    hour_angle = 2 * np.pi * index.hour.to_numpy() / 24
    weekday_angle = 2 * np.pi * index.dayofweek.to_numpy() / 7
    return np.stack([
        np.sin(hour_angle), np.cos(hour_angle),
        np.sin(weekday_angle), np.cos(weekday_angle),
    ], axis=1).astype(np.float32)


def make_origins(frame, split, stride):
    labels = frame["split"].to_numpy()
    valid = np.isfinite(frame["load_kwh"].to_numpy())
    candidates = []
    accepted = []
    for origin in range(LOOKBACK, len(frame) - HORIZON + 1):
        if frame.index[origin].hour % stride != 0:
            continue
        if not np.all(labels[origin:origin + HORIZON] == split):
            continue
        candidates.append(origin)
        if valid[origin - LOOKBACK:origin + HORIZON].all():
            accepted.append(origin)
    audit = {"candidate_windows": len(candidates), "retained_windows": len(accepted), "excluded_missing_windows": len(candidates) - len(accepted)}
    if not accepted:
        raise ValueError(f"No complete windows for {split}: {audit}")
    return np.asarray(accepted, dtype=np.int64), audit


class LoadDataset(Dataset):
    def __init__(self, frame, origins, mean, scale):
        self.origins = origins
        self.raw = frame["load_kwh"].to_numpy(dtype=np.float32)
        standardized = ((self.raw - mean) / scale).astype(np.float32)
        self.features = np.column_stack([standardized, calendar_features(frame.index)])
        self.targets = standardized

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, item):
        origin = self.origins[item]
        history = self.features[origin - LOOKBACK:origin]
        target = self.targets[origin:origin + HORIZON]
        return torch.from_numpy(history), torch.from_numpy(target)

    def actual(self):
        return np.stack([self.raw[origin:origin + HORIZON] for origin in self.origins])

    def seasonal_naive(self):
        return np.stack([self.raw[origin - LOOKBACK:origin - LOOKBACK + HORIZON] for origin in self.origins])


class LoadTransformer(nn.Module):
    def __init__(self, width=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(5, width)
        self.position = nn.Parameter(torch.empty(1, LOOKBACK, width))
        nn.init.normal_(self.position, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 2,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
        self.output = nn.Linear(LOOKBACK * width, HORIZON)

    def forward(self, history):
        encoded = self.encoder(self.embedding(history) + self.position)
        return self.output(encoded.flatten(start_dim=1))


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require equal-shaped finite arrays.")
    error = predicted - actual
    denominator = np.abs(actual).sum()
    return {
        "mae_kwh": float(np.abs(error).mean()),
        "rmse_kwh": float(np.sqrt(np.square(error).mean())),
        "wape_percent": float(100 * np.abs(error).sum() / denominator) if denominator > 0 else None,
    }
