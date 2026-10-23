"""Compare separate and shared Transformers on a fixed training-selected cohort."""

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from forecasting import HORIZON, LOOKBACK, LoadDataset, LoadTransformer, make_origins, metrics
from run_experiment import predict


SPLITS = ("train", "validation", "test")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_buildings(data_dir, batch_size, train_stride):
    manifest = json.loads((data_dir / "data_summary.json").read_text())
    if "buildings" not in manifest or len(manifest["buildings"]) < 2:
        raise ValueError("Prepare at least two buildings using prepare_data.py --num-buildings N.")
    if manifest.get("num_buildings") != len(manifest["buildings"]):
        raise ValueError("Building manifest count is inconsistent.")
    buildings = {}
    for item in manifest["buildings"]:
        building_id = item["building_id"]
        directory = data_dir / item["data_dir"]
        data_path = directory / "building.csv"
        summary = json.loads((directory / "data_summary.json").read_text())
        if summary["building_id"] != building_id:
            raise ValueError(f"Building summary ID mismatch: {building_id}")
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if digest != summary["processed_sha256"] or digest != item["processed_sha256"]:
            raise ValueError(f"Prepared data checksum mismatch: {building_id}")
        frame = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
        expected = pd.date_range("2016-01-01", "2017-12-31 23:00", freq="h")
        if not frame.index.equals(expected):
            raise ValueError(f"Unexpected hourly index: {building_id}")
        expected_splits = np.where(frame.index < "2017-01-01", "train", np.where(frame.index < "2017-07-01", "validation", "test"))
        if not np.array_equal(frame["split"].to_numpy(), expected_splits):
            raise ValueError(f"Unexpected date split: {building_id}")
        train = frame.loc[frame["split"] == "train", "load_kwh"].dropna().to_numpy()
        mean, scale = float(train.mean()), float(train.std())
        if not np.isfinite([mean, scale]).all() or scale <= 0:
            raise ValueError(f"Invalid training normalization: {building_id}")
        datasets, audits, loaders = {}, {}, {}
        for split in SPLITS:
            origins, audits[split] = make_origins(frame, split, train_stride if split == "train" else 24)
            datasets[split] = LoadDataset(frame, origins, mean, scale)
            if split != "train":
                loaders[split] = DataLoader(datasets[split], batch_size=batch_size, shuffle=False)
        buildings[building_id] = dict(directory=directory, frame=frame, summary=summary,
                                      mean=mean, scale=scale, datasets=datasets, audits=audits, loaders=loaders)
    if list(buildings) != manifest["building_ids"]:
        raise ValueError("Building manifest IDs are duplicated or inconsistent.")
    return manifest, buildings


def train_shared(buildings, arguments, seed, output_dir):
    output_dir.mkdir()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    dataset = ConcatDataset([b["datasets"]["train"] for b in buildings.values()])
    loader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=True)
    architecture = dict(width=32, heads=4, layers=2, dropout=0.1)
    model = LoadTransformer(**architecture).to(device)
    configuration = {
        "seed": seed, "device": "cpu", "architecture": architecture,
        "lookback": LOOKBACK, "horizon": HORIZON,
        "epochs": arguments.epochs, "patience": arguments.patience,
        "batch_size": arguments.batch_size, "learning_rate": arguments.learning_rate,
        "train_stride": arguments.train_stride, "training_windows": len(dataset),
        "parameters": sum(p.numel() for p in model.parameters()),
        "normalization": {key: dict(mean=b["mean"], scale=b["scale"]) for key, b in buildings.items()},
        "data": {key: b["summary"] for key, b in buildings.items()},
        "window_audit": {key: b["audits"] for key, b in buildings.items()},
        "training_objective": "MSE of per-building training-standardized targets; uniformly shuffled retained windows",
        "checkpoint_selection": "Lowest equal-building mean of validation MAE / building training std; no test selection",
        "evaluation": "Per-building midnight forecasts of next 24 hours; clip negative kWh predictions to zero",
        "versions": dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
    }
    write_json(output_dir / "config.json", configuration)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=0.01)
    objective = nn.MSELoss()
    best_score, best_epoch, optimizer_steps = float("inf"), 0, 0
    rows = []
    start = time.monotonic()
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for history, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(history.to(device)), target.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite shared training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer_steps += 1
            loss_sum += loss.item() * len(history)
            count += len(history)
        validation_mae, validation_rmse = [], []
        for b in buildings.values():
            predicted = predict(model, b["loaders"]["validation"], device, b["mean"], b["scale"])
            scores = metrics(b["datasets"]["validation"].actual(), predicted)
            validation_mae.append(scores["mae_kwh"] / b["scale"])
            validation_rmse.append(scores["rmse_kwh"] / b["scale"])
        score = float(np.mean(validation_mae))
        row = dict(epoch=epoch, train_mse_standardized=loss_sum / count,
                   validation_macro_mae_train_std=score,
                   validation_macro_rmse_train_std=float(np.mean(validation_rmse)),
                   optimizer_steps=optimizer_steps, elapsed_seconds=time.monotonic() - start)
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / "training_history.csv", index=False)
        print(json.dumps(dict(run="shared", seed=seed, **row)), flush=True)
        if score < best_score:
            best_score, best_epoch = score, epoch
            torch.save(dict(model=model.state_dict(), architecture=architecture,
                            normalization=configuration["normalization"], epoch=epoch), output_dir / "best_model.pt")
        if epoch - best_epoch >= arguments.patience:
            break
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    result_rows, predictions, horizons = [], [], []
    for building_id, b in buildings.items():
        for split in ("validation", "test"):
            d = b["datasets"][split]
            actual = d.actual()
            estimated = predict(model, b["loaders"][split], device, b["mean"], b["scale"])
            result_rows.append(dict(building_id=building_id, split=split, model="shared_transformer",
                                    forecast_days=len(d), **metrics(actual, estimated)))
            for step in range(HORIZON):
                horizons.append(dict(building_id=building_id, split=split, model="shared_transformer",
                                     horizon_hour=step + 1, **metrics(actual[:, step], estimated[:, step])))
            predictions.append(pd.DataFrame(dict(
                building_id=building_id, split=split,
                origin=np.repeat(b["frame"].index[d.origins].to_numpy(), HORIZON),
                timestamp=b["frame"].index[(d.origins[:, None] + np.arange(HORIZON)).ravel()],
                horizon_hour=np.tile(np.arange(1, HORIZON + 1), len(d)),
                actual_kwh=actual.ravel(), shared_transformer_kwh=estimated.ravel())))
    pd.DataFrame(result_rows).to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(horizons).to_csv(output_dir / "horizon_metrics.csv", index=False)
    write_json(output_dir / "result.json", dict(best_epoch=best_epoch, epochs_run=len(rows),
                                               optimizer_steps=optimizer_steps,
                                               elapsed_seconds=time.monotonic() - start, results=result_rows))


def collect_comparison(buildings, seeds, output_dir):
    """Recompute paired results from exported predictions, checking identical targets."""
    rows, aggregate_rows = [], []
    keys = ["split", "origin", "timestamp", "horizon_hour"]
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        shared = pd.read_csv(seed_dir / "shared" / "predictions.csv")
        merged_frames = []
        for building_id, b in buildings.items():
            single = pd.read_csv(seed_dir / "individual" / building_id / "predictions.csv")
            common = single.merge(shared[shared["building_id"] == building_id], on=keys,
                                  how="outer", validate="one_to_one", indicator=True, suffixes=("", "_shared"))
            if not common["_merge"].eq("both").all() or not np.array_equal(common.actual_kwh, common.actual_kwh_shared):
                raise ValueError(f"Single/shared evaluation targets differ: {building_id}")
            common["training_std_kwh"] = b["scale"]
            merged_frames.append(common)
            for split, frame in common.groupby("split", sort=False):
                for name, column in (("weekly_naive", "weekly_naive_kwh"),
                                     ("individual_transformer", "transformer_kwh"),
                                     ("shared_transformer", "shared_transformer_kwh")):
                    score = metrics(frame.actual_kwh, frame[column])
                    rows.append(dict(seed=seed, building_id=building_id, split=split, model=name,
                                     forecast_hours=len(frame), training_std_kwh=b["scale"],
                                     rmse_train_std=score["rmse_kwh"] / b["scale"],
                                     mae_train_std=score["mae_kwh"] / b["scale"], **score))
        merged = pd.concat(merged_frames, ignore_index=True)
        for split, frame in merged.groupby("split", sort=False):
            for name, column in (("weekly_naive", "weekly_naive_kwh"),
                                 ("individual_transformer", "transformer_kwh"),
                                 ("shared_transformer", "shared_transformer_kwh")):
                score = metrics(frame.actual_kwh, frame[column])
                selected = [r for r in rows if r["seed"] == seed and r["split"] == split and r["model"] == name]
                aggregate_rows.append(dict(seed=seed, split=split, model=name, buildings=len(buildings),
                                           forecast_hours=len(frame), pooled_rmse_kwh=score["rmse_kwh"],
                                           pooled_mae_kwh=score["mae_kwh"], pooled_wape_percent=score["wape_percent"],
                                           macro_rmse_train_std=float(np.mean([r["rmse_train_std"] for r in selected])),
                                           macro_mae_train_std=float(np.mean([r["mae_train_std"] for r in selected]))))
    scores = pd.DataFrame(rows)
    scores.to_csv(output_dir / "per_building_metrics.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(output_dir / "aggregate_metrics.csv", index=False)
    paired = scores.pivot(index=["seed", "building_id", "split"], columns="model", values="rmse_kwh").reset_index()
    paired.columns.name = None
    paired["shared_improvement_vs_individual_percent"] = 100 * (1 - paired.shared_transformer / paired.individual_transformer.replace(0, np.nan))
    paired["shared_improvement_vs_naive_percent"] = 100 * (1 - paired.shared_transformer / paired.weekly_naive.replace(0, np.nan))
    paired.to_csv(output_dir / "rmse_comparison.csv", index=False)
    summary = scores.groupby(["building_id", "split", "model"]).agg(
        seeds=("seed", "count"), rmse_kwh_mean=("rmse_kwh", "mean"), rmse_kwh_std=("rmse_kwh", "std"),
        mae_kwh_mean=("mae_kwh", "mean"), wape_percent_mean=("wape_percent", "mean")).reset_index()
    summary.to_csv(output_dir / "seed_summary.csv", index=False)
    print(summary[summary.split == "test"].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed_5_buildings"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multi_building_2026-10-23"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--train-stride", type=int, choices=[1, 3, 6, 12, 24], default=6)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs, patience, batch-size and learning-rate must be positive and finite")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("seeds must be unique integers in [0, 2**32)")
    manifest, buildings = load_buildings(args.data_dir, args.batch_size, args.train_stride)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is not empty; use a new directory to preserve previous experiments.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.output_dir / "source"
    source_dir.mkdir()
    source_files = ("run_multibuilding.py", "run_experiment.py", "forecasting.py", "prepare_data.py")
    for name in source_files:
        shutil.copy2(Path(__file__).with_name(name), source_dir / name)
    write_json(args.output_dir / "experiment_plan.json", {
        "question": f"Does training on {len(buildings)} preselected buildings reduce RMSE relative to one model per building?",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cohort": manifest, "primary_endpoint": "Original building test RMSE, paired by seed and identical forecast timestamps",
        "original_building_id": manifest["building_ids"][0],
        "secondary_endpoints": "Per-building RMSE; equal-building mean RMSE / training std; pooled kWh RMSE",
        "selection": "Building selection uses 2016 only; checkpoint selection uses 2017 Jan-Jun only",
        "budget": "Same max epochs, optimizer and architecture; shared has more updates per epoch than each individual model",
        "limitations": f"Exploratory follow-up: original building test period previously examined; {len(args.seeds)} seeds are not independent buildings; no hyperparameter search",
        "code_sha256": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                        for name in source_files},
    })
    script = source_dir / "run_experiment.py"
    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir()
        (seed_dir / "individual").mkdir()
        for building_id, b in buildings.items():
            command = [sys.executable, str(script), "--data-dir", str(b["directory"]),
                       "--output-dir", str(seed_dir / "individual" / building_id),
                       "--epochs", str(args.epochs), "--patience", str(args.patience),
                       "--batch-size", str(args.batch_size), "--learning-rate", str(args.learning_rate),
                       "--train-stride", str(args.train_stride), "--seed", str(seed), "--device", "cpu"]
            print(json.dumps(dict(run="individual", seed=seed, building_id=building_id, command=command)), flush=True)
            with (seed_dir / "individual" / f"{building_id}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        train_shared(buildings, args, seed, seed_dir / "shared")
    collect_comparison(buildings, args.seeds, args.output_dir)
    write_json(args.output_dir / "completion.json", {"status": "complete", "seeds": args.seeds, "building_ids": list(buildings)})


if __name__ == "__main__":
    main()
