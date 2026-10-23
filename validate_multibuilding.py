"""Independently verify a completed multi-building experiment without training.

Run after run_multibuilding.py finishes. Only verification.json in --run-dir is
written, and only after every check succeeds. Predictions are rounded float32
values in CSV, so comparisons allow small serialization/inference differences.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forecasting import LoadDataset, LoadTransformer


LOOKBACK, HORIZON = 168, 24
KEYS = ["split", "origin", "timestamp", "horizon_hour"]
METRIC_RTOL, METRIC_ATOL = 2e-6, 2e-6
PREDICTION_RTOL, PREDICTION_ATOL = 2e-5, 2e-4


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual, expected, label, rtol=METRIC_RTOL, atol=METRIC_ATOL):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, f"{label}: shape mismatch")
    require(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True),
            f"{label}: values differ (maximum absolute difference "
            f"{np.nanmax(np.abs(actual - expected)) if actual.size else 0})")


def independent_scores(actual, predicted):
    """Do not call the experiment's metrics function."""
    actual, predicted = np.asarray(actual, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    require(actual.shape == predicted.shape and actual.size > 0, "Empty or mismatched metric inputs")
    require(np.isfinite(actual).all() and np.isfinite(predicted).all(), "Nonfinite prediction or target")
    residual = actual - predicted
    absolute_sum = np.linalg.norm(residual, ord=1)
    return dict(mae_kwh=float(absolute_sum / len(actual)),
                rmse_kwh=float(np.linalg.norm(residual) / np.sqrt(len(actual))),
                wape_percent=float(100 * absolute_sum / np.linalg.norm(actual, ord=1))
                if np.any(actual) else np.nan)


def compare_table(actual, expected, keys, label):
    """Check complete row coverage before comparing numeric fields."""
    expected = pd.DataFrame(expected)
    require(not actual.duplicated(keys).any(), f"{label}: duplicate rows")
    require(not expected.duplicated(keys).any(), f"{label}: duplicate expected rows")
    columns = [c for c in expected if c not in keys]
    require(set(keys + columns).issubset(actual.columns), f"{label}: missing columns")
    paired = expected.merge(actual[keys + columns], on=keys, how="outer",
                            suffixes=("_expected", "_actual"), indicator=True, validate="one_to_one")
    require(paired["_merge"].eq("both").all(), f"{label}: missing or unexpected rows")
    for column in columns:
        close(paired[column + "_actual"], paired[column + "_expected"], f"{label}/{column}")


def independent_windows(frame, split, stride):
    """Derive complete windows from dates and finite values, independent of make_origins."""
    starts = {"train": pd.Timestamp("2016-01-01"), "validation": pd.Timestamp("2017-01-01"),
              "test": pd.Timestamp("2017-07-01")}
    ends = {"train": pd.Timestamp("2017-01-01"), "validation": pd.Timestamp("2017-07-01"),
            "test": pd.Timestamp("2018-01-01")}
    timestamps = frame.index.to_numpy(dtype="datetime64[ns]")
    candidates = np.flatnonzero((timestamps >= starts[split].to_datetime64()) &
                               (timestamps + np.timedelta64(23, "h") < ends[split].to_datetime64()) &
                               (frame.index.hour % stride == 0))
    candidates = candidates[candidates >= LOOKBACK]
    missing = ~np.isfinite(frame.load_kwh.to_numpy())
    cumulative = np.r_[0, np.cumsum(missing)]
    origins = candidates[(cumulative[candidates + HORIZON] - cumulative[candidates - LOOKBACK]) == 0]
    require(len(origins) > 0, f"No retained {split} windows")
    return origins, dict(candidate_windows=len(candidates), retained_windows=len(origins),
                         excluded_missing_windows=len(candidates) - len(origins))


def expected_evaluation(frame, origins):
    values = frame.load_kwh.to_numpy(dtype=np.float32)
    rows = []
    for split in ("validation", "test"):
        starts = origins[split]
        positions = (starts[:, None] + np.arange(HORIZON)).ravel()
        rows.append(pd.DataFrame(dict(split=split, origin=np.repeat(frame.index[starts], HORIZON),
                                      timestamp=frame.index[positions],
                                      horizon_hour=np.tile(np.arange(1, HORIZON + 1), len(starts)),
                                      actual_kwh=values[positions],
                                      weekly_naive_kwh=values[positions - LOOKBACK])))
    return pd.concat(rows, ignore_index=True)


def load_prediction(path):
    result = pd.read_csv(path, parse_dates=["origin", "timestamp"])
    require(not result.duplicated(KEYS + (["building_id"] if "building_id" in result else [])).any(),
            f"{path}: duplicate forecast rows")
    group_keys = ["building_id", "split"] if "building_id" in result else ["split"]
    require(not result.duplicated(group_keys + ["timestamp"]).any(), f"{path}: overlapping target hours")
    return result


def check_training(directory, config, training_windows, shared=False):
    history = pd.read_csv(directory / "training_history.csv")
    result = read_json(directory / "result.json")
    require(history.epoch.tolist() == list(range(1, len(history) + 1)), f"{directory}: epoch sequence")
    require(0 < len(history) <= config["epochs"], f"{directory}: epoch limit")
    require(result["epochs_run"] == len(history), f"{directory}: epochs_run mismatch")
    selection = "validation_macro_mae_train_std" if shared else "mae_kwh"
    require(np.isfinite(history[selection]).all(), f"{directory}: nonfinite validation score")
    best_epoch = int(history.loc[history[selection].idxmin(), "epoch"])
    require(result["best_epoch"] == best_epoch, f"{directory}: wrong validation-selected epoch")
    steps_per_epoch = (training_windows + config["batch_size"] - 1) // config["batch_size"]
    if shared:
        require(history.optimizer_steps.tolist() == (history.epoch * steps_per_epoch).tolist(),
                f"{directory}: optimizer step sequence")
        require(result["optimizer_steps"] == len(history) * steps_per_epoch,
                f"{directory}: optimizer step total")
    return result, dict(best_epoch=best_epoch, epochs_run=len(history),
                        training_windows=training_windows, optimizer_steps=len(history) * steps_per_epoch)


def verify(data_dir, run_dir, reference_run, skip_reference=False):
    completion = read_json(run_dir / "completion.json")
    require(completion["status"] == "complete", "Experiment is incomplete")
    manifest = read_json(data_dir / "data_summary.json")
    plan = read_json(run_dir / "experiment_plan.json")
    source_hashes = plan["code_sha256"]
    require(set(source_hashes) == {"run_multibuilding.py", "run_experiment.py", "forecasting.py", "prepare_data.py"},
            "Experiment plan must record all four source snapshots")
    for filename, expected_hash in source_hashes.items():
        snapshot = run_dir / "source" / filename
        require(snapshot.is_file(), f"Missing source snapshot: {snapshot}")
        require(hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected_hash,
                f"Source snapshot checksum differs from experiment plan: {filename}")
    require(plan["cohort"] == manifest, "Prepared cohort differs from experiment plan")
    buildings, seeds = manifest["building_ids"], completion["seeds"]
    require(buildings == completion["building_ids"], "Completion building IDs differ")
    require(seeds == plan["arguments"]["seeds"] and len(set(seeds)) == len(seeds), "Seed mismatch")
    require(len(buildings) == manifest["num_buildings"] == len(set(buildings)), "Building count mismatch")
    quality = pd.read_csv(data_dir / "training_building_quality.csv")
    require(not quality.building_id.duplicated().any(), "Duplicate building quality entries")
    eligible = ((quality.missing_fraction <= .005) & (quality.nonpositive_fraction <= .01) &
                (quality.mean_kwh > 0) & (quality.std_kwh > 0) &
                np.isfinite(quality.mean_kwh) & np.isfinite(quality.std_kwh))
    require(np.array_equal(quality.eligible, eligible), "Quality eligibility differs from published rule")
    ranking = quality[eligible].sort_values(["missing_fraction", "nonpositive_fraction", "building_id"])
    require(ranking.building_id.head(len(buildings)).tolist() == buildings, "Cohort is not first N eligible buildings")
    require(len(ranking) == manifest["eligible_training_buildings"], "Eligible cohort count mismatch")

    prepared, data_report = {}, {}
    expected_index = pd.date_range("2016-01-01", "2017-12-31 23:00", freq="h")
    labels = np.where(expected_index < "2017-01-01", "train",
                      np.where(expected_index < "2017-07-01", "validation", "test"))
    for item in manifest["buildings"]:
        bid = item["building_id"]
        directory = data_dir / item["data_dir"]
        path = directory / "building.csv"
        summary = read_json(directory / "data_summary.json")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item["processed_sha256"] == summary["processed_sha256"],
                f"{bid}: data checksum mismatch")
        frame = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        require(frame.index.equals(expected_index), f"{bid}: hourly timestamp range")
        require(np.array_equal(frame.split, labels), f"{bid}: temporal split boundaries")
        training = frame.loc[frame.index < "2017-01-01", "load_kwh"].dropna().to_numpy()
        mean, scale = float(np.mean(training)), float(np.std(training, ddof=0))
        require(np.isfinite([mean, scale]).all() and scale > 0, f"{bid}: invalid training statistics")
        origins, audits = {}, {}
        for split in ("train", "validation", "test"):
            origins[split], audits[split] = independent_windows(
                frame, split, plan["arguments"]["train_stride"] if split == "train" else 24)
        evaluation = expected_evaluation(frame, origins)

        # Change every finite value at/after first test issue time. Training stats,
        # that forecast's inputs and its lag-week baseline must remain unchanged.
        first = int(origins["test"][0])
        changed = frame.copy()
        changed.iloc[first:, changed.columns.get_loc("load_kwh")] = changed.load_kwh.iloc[first:] * 7 + 12345
        changed_training = changed.loc[changed.index < "2017-01-01", "load_kwh"].dropna().to_numpy()
        close([changed_training.mean(), changed_training.std(ddof=0)], [mean, scale],
              f"{bid}: future values changed normalization", rtol=0, atol=0)
        before, after = [LoadDataset(f, np.array([first]), mean, scale) for f in (frame, changed)]
        require(torch.equal(before[0][0], after[0][0]), f"{bid}: future values leak into first test input")
        require(np.array_equal(before.seasonal_naive(), after.seasonal_naive()), f"{bid}: future baseline leak")
        require(not torch.equal(before[0][1], after[0][1]), f"{bid}: future perturbation was ineffective")
        prepared[bid] = dict(frame=frame, mean=mean, scale=scale, origins=origins, audits=audits,
                             evaluation=evaluation, summary=summary)
        data_report[bid] = dict(train_mean_kwh=mean, train_std_kwh=scale,
                                forecast_days={split: len(origins[split]) for split in ("validation", "test")},
                                future_perturbation="passed")
    require(list(prepared) == buildings, "Manifest building entries differ from ID list")

    all_scores, aggregate_scores, training_report, restored = [], [], {}, []
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        shared_dir = seed_dir / "shared"
        shared_config = read_json(shared_dir / "config.json")
        require(shared_config["seed"] == seed, f"Seed {seed}: wrong shared seed")
        require(set(shared_config["normalization"]) == set(buildings), f"Seed {seed}: normalization building IDs")
        require(shared_config["lookback"] == LOOKBACK and shared_config["horizon"] == HORIZON,
                f"Seed {seed}: incompatible forecast dimensions")
        shared_predictions = load_prediction(shared_dir / "predictions.csv")
        require(set(shared_predictions.building_id) == set(buildings), f"Seed {seed}: shared prediction cohort")
        total_windows = sum(len(b["origins"]["train"]) for b in prepared.values())
        require(shared_config["training_windows"] == total_windows, f"Seed {seed}: total training windows")
        shared_result, shared_training = check_training(shared_dir, shared_config, total_windows, shared=True)
        training_report[str(seed)] = dict(shared=shared_training, individual={})
        checkpoint = torch.load(shared_dir / "best_model.pt", map_location="cpu", weights_only=True)
        require(checkpoint["architecture"] == shared_config["architecture"], f"Seed {seed}: checkpoint architecture")
        require(checkpoint["epoch"] == shared_result["best_epoch"], f"Seed {seed}: checkpoint epoch")
        require(checkpoint["normalization"] == shared_config["normalization"], f"Seed {seed}: checkpoint normalization")
        model = LoadTransformer(**checkpoint["architecture"])
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        require(sum(p.numel() for p in model.parameters()) == shared_config["parameters"], f"Seed {seed}: parameter count")
        shared_metric_rows, shared_horizon_rows, joined_buildings = [], [], []
        for bid, b in prepared.items():
            single_dir = seed_dir / "individual" / bid
            config = read_json(single_dir / "config.json")
            require(config["seed"] == seed, f"{seed}/{bid}: wrong individual seed")
            for key in ("epochs", "patience", "batch_size", "learning_rate", "train_stride", "architecture", "lookback", "horizon"):
                require(config[key] == shared_config[key], f"{seed}/{bid}: unequal {key}")
                if key in plan["arguments"]:
                    require(config[key] == plan["arguments"][key], f"{seed}/{bid}: plan {key} differs")
            require(config["window_audit"] == b["audits"] == shared_config["window_audit"][bid],
                    f"{seed}/{bid}: window audit mismatch")
            require(config["data"] == b["summary"] == shared_config["data"][bid], f"{seed}/{bid}: input metadata")
            close([config["train_mean_kwh"], config["train_std_kwh"],
                   shared_config["normalization"][bid]["mean"], shared_config["normalization"][bid]["scale"]],
                  [b["mean"], b["scale"], b["mean"], b["scale"]], f"{seed}/{bid}: training normalization",
                  rtol=1e-12, atol=1e-12)
            single_result, training_record = check_training(single_dir, config, len(b["origins"]["train"]))
            training_report[str(seed)]["individual"][bid] = training_record
            single = load_prediction(single_dir / "predictions.csv")
            shared = shared_predictions[shared_predictions.building_id == bid].drop(columns="building_id")
            compare_table(single, b["evaluation"], KEYS, f"{seed}/{bid}: actuals and weekly baseline")
            compare_table(shared, b["evaluation"].drop(columns="weekly_naive_kwh"), KEYS,
                          f"{seed}/{bid}: shared actuals")
            combined = single.merge(shared, on=KEYS, how="outer", validate="one_to_one",
                                    suffixes=("", "_shared"), indicator=True)
            require(combined["_merge"].eq("both").all(), f"{seed}/{bid}: paired timestamps mismatch")
            close(combined.actual_kwh, combined.actual_kwh_shared, f"{seed}/{bid}: paired actuals", rtol=0, atol=0)
            require((combined[["transformer_kwh", "shared_transformer_kwh"]] >= 0).all().all(),
                    f"{seed}/{bid}: negative clipped predictions")
            combined["scale"] = b["scale"]
            joined_buildings.append(combined)
            individual_metric_rows, individual_horizon_rows = [], []
            for split in ("validation", "test"):
                part = combined[combined.split == split]
                for name, column, local_name in (("weekly_naive", "weekly_naive_kwh", "weekly_naive"),
                                                  ("individual_transformer", "transformer_kwh", "transformer"),
                                                  ("shared_transformer", "shared_transformer_kwh", "shared_transformer")):
                    scores = independent_scores(part.actual_kwh, part[column])
                    standardized = independent_scores(part.actual_kwh / b["scale"], part[column] / b["scale"])
                    all_scores.append(dict(seed=seed, building_id=bid, split=split, model=name,
                                           forecast_hours=len(part), training_std_kwh=b["scale"],
                                           rmse_train_std=standardized["rmse_kwh"], mae_train_std=standardized["mae_kwh"], **scores))
                    row = dict(split=split, model=local_name, forecast_days=len(part) // HORIZON, **scores)
                    target_metrics = shared_metric_rows if name == "shared_transformer" else individual_metric_rows
                    target_horizons = shared_horizon_rows if name == "shared_transformer" else individual_horizon_rows
                    if name == "shared_transformer":
                        row["building_id"] = bid
                    target_metrics.append(row)
                    for hour, subset in part.groupby("horizon_hour"):
                        row = dict(split=split, model=local_name, horizon_hour=int(hour),
                                   **independent_scores(subset.actual_kwh, subset[column]))
                        if name == "shared_transformer":
                            row["building_id"] = bid
                        target_horizons.append(row)
            compare_table(pd.read_csv(single_dir / "metrics.csv"), individual_metric_rows, ["split", "model"], f"{seed}/{bid}: metrics")
            compare_table(pd.DataFrame(single_result["results"]), individual_metric_rows, ["split", "model"], f"{seed}/{bid}: result JSON")
            compare_table(pd.read_csv(single_dir / "horizon_metrics.csv"), individual_horizon_rows,
                          ["split", "model", "horizon_hour"], f"{seed}/{bid}: horizon metrics")

            # Construct checkpoint inputs independently, including float32 order.
            first_two = b["origins"]["test"][:2]
            require(len(first_two) == 2, f"{bid}: fewer than two test days")
            inputs = []
            for origin in first_two:
                dates = b["frame"].index[origin - LOOKBACK:origin]
                loads = b["frame"].load_kwh.iloc[origin - LOOKBACK:origin].to_numpy(dtype=np.float32)
                load_feature = ((loads - b["mean"]) / b["scale"]).astype(np.float32)
                calendar = np.stack([np.sin(2 * np.pi * dates.hour / 24), np.cos(2 * np.pi * dates.hour / 24),
                                     np.sin(2 * np.pi * dates.dayofweek / 7), np.cos(2 * np.pi * dates.dayofweek / 7)], axis=1).astype(np.float32)
                inputs.append(np.column_stack([load_feature, calendar]))
            with torch.inference_mode():
                predicted = model(torch.from_numpy(np.stack(inputs))).numpy()
            restored_predictions = np.maximum(predicted * b["scale"] + b["mean"], 0).ravel()
            saved = shared[(shared.split == "test") & shared.origin.isin(b["frame"].index[first_two])].sort_values(["origin", "horizon_hour"])
            close(restored_predictions, saved.shared_transformer_kwh, f"{seed}/{bid}: restored checkpoint",
                  rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL)
            restored.append(dict(seed=seed, building_id=bid, forecast_hours=len(saved),
                                 max_absolute_difference_kwh=float(np.max(np.abs(restored_predictions - saved.shared_transformer_kwh)))))
        compare_table(pd.read_csv(shared_dir / "metrics.csv"), shared_metric_rows,
                      ["building_id", "split", "model"], f"{seed}: shared metrics")
        compare_table(pd.DataFrame(shared_result["results"]), shared_metric_rows,
                      ["building_id", "split", "model"], f"{seed}: shared result JSON")
        compare_table(pd.read_csv(shared_dir / "horizon_metrics.csv"), shared_horizon_rows,
                      ["building_id", "split", "model", "horizon_hour"], f"{seed}: shared horizon metrics")
        pooled = pd.concat(joined_buildings, ignore_index=True)
        for split in ("validation", "test"):
            part = pooled[pooled.split == split]
            for name, column in (("weekly_naive", "weekly_naive_kwh"), ("individual_transformer", "transformer_kwh"),
                                 ("shared_transformer", "shared_transformer_kwh")):
                score = independent_scores(part.actual_kwh, part[column])
                members = [r for r in all_scores if r["seed"] == seed and r["split"] == split and r["model"] == name]
                require(len(members) == len(buildings), "Missing macro-average building")
                aggregate_scores.append(dict(seed=seed, split=split, model=name, buildings=len(buildings), forecast_hours=len(part),
                                             pooled_rmse_kwh=score["rmse_kwh"], pooled_mae_kwh=score["mae_kwh"],
                                             pooled_wape_percent=score["wape_percent"],
                                             macro_rmse_train_std=sum(r["rmse_train_std"] for r in members) / len(members),
                                             macro_mae_train_std=sum(r["mae_train_std"] for r in members) / len(members)))

    compare_table(pd.read_csv(run_dir / "per_building_metrics.csv"), all_scores,
                  ["seed", "building_id", "split", "model"], "Per-building comparison metrics")
    compare_table(pd.read_csv(run_dir / "aggregate_metrics.csv"), aggregate_scores,
                  ["seed", "split", "model"], "Aggregate comparison metrics")
    expected_summary, expected_comparison = [], []
    scores_frame = pd.DataFrame(all_scores)
    for (bid, split, name), group in scores_frame.groupby(["building_id", "split", "model"]):
        expected_summary.append(dict(building_id=bid, split=split, model=name, seeds=len(group),
                                     rmse_kwh_mean=np.mean(group.rmse_kwh),
                                     rmse_kwh_std=np.std(group.rmse_kwh, ddof=1) if len(group) > 1 else np.nan,
                                     mae_kwh_mean=np.mean(group.mae_kwh), wape_percent_mean=np.mean(group.wape_percent)))
    for (seed, bid, split), group in scores_frame.groupby(["seed", "building_id", "split"]):
        values = dict(zip(group.model, group.rmse_kwh))
        expected_comparison.append(dict(seed=seed, building_id=bid, split=split, **values,
                                        shared_improvement_vs_individual_percent=100 * (1 - values["shared_transformer"] / values["individual_transformer"]) if values["individual_transformer"] else np.nan,
                                        shared_improvement_vs_naive_percent=100 * (1 - values["shared_transformer"] / values["weekly_naive"]) if values["weekly_naive"] else np.nan))
    compare_table(pd.read_csv(run_dir / "seed_summary.csv"), expected_summary, ["building_id", "split", "model"], "Seed summary")
    compare_table(pd.read_csv(run_dir / "rmse_comparison.csv"), expected_comparison, ["seed", "building_id", "split"], "RMSE comparison")

    if skip_reference:
        reference_equality = dict(status="not_applicable", reason="Explicit --skip-reference; original-run reproduction was not checked")
    else:
        reference_config = read_json(reference_run / "config.json")
        reference_id = reference_config["data"]["building_id"]
        require(42 in seeds and reference_id in buildings,
                "Seed 42 and original building are required for reference verification; use --skip-reference for other experiments")
        rerun = run_dir / "seed_42" / "individual" / reference_id
        rerun_config = read_json(rerun / "config.json")
        for key in ("seed", "epochs", "patience", "batch_size", "learning_rate", "train_stride", "device", "architecture",
                    "lookback", "horizon", "train_mean_kwh", "train_std_kwh", "parameters", "window_audit", "data"):
            require(rerun_config[key] == reference_config[key],
                    f"Seed 42 reference/config {key} differs; use --skip-reference for intentionally different experiments")
        reference_equality = {"status": "passed"}
        for filename, keys in (("predictions.csv", KEYS), ("metrics.csv", ["split", "model"]),
                               ("horizon_metrics.csv", ["split", "model", "horizon_hour"])):
            old, new = pd.read_csv(reference_run / filename), pd.read_csv(rerun / filename)
            compare_table(new, old, keys, f"Seed 42 reference/{filename}")
            reference_equality[filename] = dict(byte_identical=(reference_run / filename).read_bytes() == (rerun / filename).read_bytes(),
                                                numerically_equal=True, rows=len(old))
        old_result, new_result = read_json(reference_run / "result.json"), read_json(rerun / "result.json")
        require(old_result["best_epoch"] == new_result["best_epoch"] and old_result["epochs_run"] == new_result["epochs_run"],
                "Seed 42 reference training selection differs")
        compare_table(pd.DataFrame(new_result["results"]), old_result["results"], ["split", "model"], "Seed 42 reference/result JSON")
    return dict(status="passed", verified_at_utc=datetime.now(timezone.utc).isoformat(),
                data_dir=str(data_dir), run_dir=str(run_dir), reference_run=str(reference_run), seeds=seeds,
                building_ids=buildings, cohort_selection="first N eligible buildings in training-only quality ranking",
                checks=["source snapshot checksums against experiment plan", "processed data checksums and exact date splits", "training-only normalization with ddof=0",
                        "complete independent window audits and unique evaluation targets", "future-value perturbation",
                        "individual/shared actuals and source-derived weekly baseline", "all validation/test metrics and horizon metrics",
                        "raw, standardized, equal-building macro and pooled metrics", "seed summaries and RMSE comparisons",
                        "validation-selected checkpoints and optimizer step counts", "shared checkpoint restoration for every seed/building"]
                        + ([] if skip_reference else ["seed 42 original-building reference reproduction"]),
                source_code_sha256=source_hashes,
                numeric_tolerances=dict(metric_rtol=METRIC_RTOL, metric_atol=METRIC_ATOL,
                                        restored_prediction_rtol=PREDICTION_RTOL, restored_prediction_atol=PREDICTION_ATOL),
                data=data_report, training=training_report, shared_checkpoint_restoration=restored,
                original_seed42_reference=reference_equality, per_building_metric_rows=len(all_scores),
                aggregate_metric_rows=len(aggregate_scores), retrained=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed_5_buildings"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/multi_building_2026-10-23"))
    parser.add_argument("--reference-run", type=Path, default=Path("outputs/first_run"))
    parser.add_argument("--skip-reference", action="store_true",
                        help="Skip original seed-42 reproduction for alternate seeds, buildings or training settings; record not_applicable")
    arguments = parser.parse_args()
    if not (arguments.run_dir / "completion.json").exists():
        parser.error("completion.json is absent; wait for the experiment to finish before verification")
    report = verify(arguments.data_dir, arguments.run_dir, arguments.reference_run, arguments.skip_reference)
    destination = arguments.run_dir / "verification.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], buildings=len(report["building_ids"]), seeds=report["seeds"],
                          verification=str(destination)), ensure_ascii=False))


if __name__ == "__main__":
    main()
