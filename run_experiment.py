import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from forecasting import HORIZON, LOOKBACK, LoadDataset, LoadTransformer, make_origins, metrics


def predict(model, loader, device, mean, scale):
    model.eval()
    batches = []
    with torch.inference_mode():
        for history, _ in loader:
            batches.append(model(history.to(device)).cpu().numpy())
    raw = np.concatenate(batches) * scale + mean
    return np.maximum(raw, 0)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate one small Transformer against weekly persistence.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/first_run"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--train-stride", type=int, choices=[1, 3, 6, 12, 24], default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.patience < 1 or arguments.batch_size < 1 or arguments.learning_rate <= 0:
        parser.error("epochs, patience, batch size and learning rate must be positive")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    if any(arguments.output_dir.iterdir()):
        raise ValueError("Output directory is not empty. Use a new run directory to preserve previous experiments.")
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.set_num_threads(4)
    if arguments.device == "cpu":
        torch.use_deterministic_algorithms(True)
    device = torch.device(arguments.device)
    frame = pd.read_csv(arguments.data_dir / "building.csv", index_col="timestamp", parse_dates=True)
    if not frame.index.equals(pd.date_range(frame.index[0], frame.index[-1], freq="h")):
        raise ValueError("Input timestamps must be unique, sorted and hourly.")
    data_summary = json.loads((arguments.data_dir / "data_summary.json").read_text())
    with (arguments.data_dir / "building.csv").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != data_summary["processed_sha256"]:
        raise ValueError("Prepared data changed after preparation; run prepare_data.py again.")
    training_values = frame.loc[frame["split"] == "train", "load_kwh"].dropna().to_numpy()
    mean, scale = float(training_values.mean()), float(training_values.std())
    if scale <= 0:
        raise ValueError("Training load has zero variation.")
    datasets, audits, loaders = {}, {}, {}
    for split in ["train", "validation", "test"]:
        stride = arguments.train_stride if split == "train" else 24
        origins, audits[split] = make_origins(frame, split, stride)
        datasets[split] = LoadDataset(frame, origins, mean, scale)
        loaders[split] = DataLoader(datasets[split], batch_size=arguments.batch_size, shuffle=split == "train")
    architecture = {"width": 32, "heads": 4, "layers": 2, "dropout": 0.1}
    model = LoadTransformer(**architecture).to(device)
    configuration = {
        **{name: str(value) if isinstance(value, Path) else value for name, value in vars(arguments).items()},
        "architecture": architecture, "lookback": LOOKBACK, "horizon": HORIZON,
        "train_mean_kwh": mean, "train_std_kwh": scale,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "evaluation": "Midnight issue times, next 24 hours; no overlapping target days; negative predictions clipped at zero",
        "window_audit": audits, "data": data_summary,
    }
    (arguments.output_dir / "config.json").write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"building": data_summary["building_id"], "device": str(device), "parameters": configuration["parameters"], "windows": audits}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=0.01)
    objective = nn.MSELoss()
    best_mae = float("inf")
    best_epoch = 0
    history_rows = []
    start_time = time.monotonic()
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        squared_loss = 0.0
        examples = 0
        for history, target in loaders["train"]:
            history, target = history.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(history), target)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            squared_loss += loss.item() * len(history)
            examples += len(history)
        validation_prediction = predict(model, loaders["validation"], device, mean, scale)
        validation_metrics = metrics(datasets["validation"].actual(), validation_prediction)
        row = {"epoch": epoch, "train_mse_standardized": squared_loss / examples, **validation_metrics, "elapsed_seconds": time.monotonic() - start_time}
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(arguments.output_dir / "training_history.csv", index=False)
        print(json.dumps(row), flush=True)
        if validation_metrics["mae_kwh"] < best_mae:
            best_mae, best_epoch = validation_metrics["mae_kwh"], epoch
            torch.save({"model": model.state_dict(), "architecture": architecture, "mean": mean, "scale": scale, "epoch": epoch}, arguments.output_dir / "best_model.pt")
        if epoch - best_epoch >= arguments.patience:
            break
    checkpoint = torch.load(arguments.output_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    results, prediction_rows, horizon_rows = [], [], []
    for split in ["validation", "test"]:
        dataset = datasets[split]
        actual = dataset.actual()
        predictions = {"weekly_naive": dataset.seasonal_naive(), "transformer": predict(model, loaders[split], device, mean, scale)}
        for name, prediction in predictions.items():
            results.append({"split": split, "model": name, "forecast_days": len(dataset), **metrics(actual, prediction)})
            for step in range(HORIZON):
                horizon_rows.append({"split": split, "model": name, "horizon_hour": step + 1, **metrics(actual[:, step], prediction[:, step])})
        for sample, origin in enumerate(dataset.origins):
            for step in range(HORIZON):
                prediction_rows.append({
                    "split": split, "origin": frame.index[origin], "timestamp": frame.index[origin + step],
                    "horizon_hour": step + 1, "actual_kwh": actual[sample, step],
                    "weekly_naive_kwh": predictions["weekly_naive"][sample, step],
                    "transformer_kwh": predictions["transformer"][sample, step],
                })
    pd.DataFrame(results).to_csv(arguments.output_dir / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(arguments.output_dir / "predictions.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(arguments.output_dir / "horizon_metrics.csv", index=False)
    completion = {"best_epoch": best_epoch, "epochs_run": len(history_rows), "elapsed_seconds": time.monotonic() - start_time, "results": results}
    (arguments.output_dir / "result.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
