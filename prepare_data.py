import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd


SOURCE_ROOT = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data"
ELECTRICITY_SHA256 = "039d909d8981e2d69eaeb366144e6ab7e84fa5e7e216aee42bddd95384a66418"
TRAIN_END = pd.Timestamp("2017-01-01")
VALIDATION_END = pd.Timestamp("2017-07-01")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_file(url, destination):
    destination = Path(destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    request = urllib.request.Request(url, headers={"User-Agent": "building-load-course-project"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)


def write_building(frame, building, output_dir, metadata, digest, metadata_digest, eligible_count, explicit):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.rename(columns={building: "load_kwh"}).copy()
    invalid = ~np.isfinite(frame["load_kwh"]) | (frame["load_kwh"] < 0)
    frame.loc[invalid, "load_kwh"] = np.nan
    frame["split"] = np.where(frame.index < TRAIN_END, "train", np.where(frame.index < VALIDATION_END, "validation", "test"))
    frame.to_csv(output_dir / "building.csv")
    frame["load_kwh"].resample("MS").agg(["count", "mean", "std", "min", "max"]).to_csv(output_dir / "monthly_load_summary.csv")
    matching = metadata.loc[metadata["building_id"] == building]
    summary = {
        "dataset": "Building Data Genome 2 (raw electricity)",
        "source": f"{SOURCE_ROOT}/meters/raw/electricity.csv",
        "raw_sha256": digest,
        "metadata_sha256": metadata_digest,
        "processed_sha256": file_hash(output_dir / "building.csv"),
        "building_id": building,
        "building_metadata": json.loads(matching.to_json(orient="records")),
        "unit": "kWh per hourly interval",
        "time_basis": "Source local-time hourly labels, retained without timezone conversion",
        "selection_rule": "Explicit ID" if explicit else "Education/office; 2016 missing <=0.5%, nonpositive <=1%, positive mean/std; sort missing, nonpositive, ID",
        "eligible_training_buildings": eligible_count,
        "processing": "Negative/nonfinite readings become missing. No interpolation, outlier clipping, or zero removal. Windows containing missing values are excluded at experiment time.",
        "splits": {},
    }
    for name, subset in frame.groupby("split", sort=False):
        summary["splits"][name] = {
            "start": str(subset.index[0]), "end": str(subset.index[-1]),
            "hours": len(subset), "missing_hours": int(subset["load_kwh"].isna().sum()),
            "zero_hours": int(subset["load_kwh"].eq(0).sum()),
            "mean_kwh": float(subset["load_kwh"].mean()),
        }
    (output_dir / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare BDG2 buildings without using test data for selection.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--building", help="Explicit building ID; omit for training-only quality selection.")
    selection.add_argument("--num-buildings", type=int, default=1,
                           help="Select the first N buildings in the training-only quality ranking (default: 1).")
    parser.add_argument("--download", action="store_true")
    arguments = parser.parse_args()
    if arguments.num_buildings < 1:
        parser.error("--num-buildings must be a positive integer.")
    if arguments.num_buildings > 1 and (arguments.output_dir / "building.csv").exists():
        parser.error("The output directory contains single-building data; choose a separate --output-dir for multiple buildings.")
    existing_summary_path = arguments.output_dir / "data_summary.json"
    if arguments.num_buildings == 1 and existing_summary_path.exists():
        existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
        if isinstance(existing_summary, dict) and "buildings" in existing_summary:
            parser.error("The output directory contains multiple-building data; choose a separate --output-dir for one building.")
    electricity_path = arguments.raw_dir / "electricity.csv"
    metadata_path = arguments.raw_dir / "metadata.csv"
    if arguments.download:
        download_file(f"{SOURCE_ROOT}/meters/raw/electricity.csv", electricity_path)
        download_file(f"{SOURCE_ROOT}/metadata/metadata.csv", metadata_path)
    digest = file_hash(electricity_path)
    if digest != ELECTRICITY_SHA256:
        raise ValueError("Raw electricity checksum differs from the verified BDG2 snapshot; inspect the download.")
    training = pd.read_csv(electricity_path, nrows=8784, index_col="timestamp", parse_dates=True)
    expected_train = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
    if not training.index.equals(expected_train):
        raise ValueError("Expected exactly the complete 2016 hourly index for building selection.")
    candidates = training.filter(regex="_(education|office)_").replace([np.inf, -np.inf], np.nan)
    quality = pd.DataFrame({
        "missing_fraction": candidates.isna().mean(),
        "nonpositive_fraction": candidates.le(0).mean(),
        "mean_kwh": candidates.mean(),
        "std_kwh": candidates.std(),
    })
    quality["eligible"] = (
        (quality["missing_fraction"] <= 0.005)
        & (quality["nonpositive_fraction"] <= 0.01)
        & (quality["mean_kwh"] > 0)
        & (quality["std_kwh"] > 0)
        & np.isfinite(quality["mean_kwh"])
        & np.isfinite(quality["std_kwh"])
    )
    quality.index.name = "building_id"
    quality = quality.sort_values(["missing_fraction", "nonpositive_fraction", "building_id"])
    eligible = quality[quality["eligible"]]
    if arguments.building is not None:
        if arguments.building not in training.columns:
            parser.error(f"Unknown building ID: {arguments.building}")
        buildings = [arguments.building]
    else:
        if arguments.num_buildings > len(eligible):
            parser.error(f"Requested {arguments.num_buildings} buildings, but only {len(eligible)} pass the training-data quality rule.")
        buildings = eligible.index[:arguments.num_buildings].tolist()
    frame = pd.read_csv(electricity_path, usecols=["timestamp", *buildings], parse_dates=["timestamp"])
    frame = frame.set_index("timestamp")
    expected_index = pd.date_range("2016-01-01", "2017-12-31 23:00", freq="h")
    if not frame.index.equals(expected_index):
        raise ValueError("Unexpected duplicate, missing, or unordered timestamps in the raw data.")
    metadata = pd.read_csv(metadata_path)
    metadata_digest = file_hash(metadata_path)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    quality.to_csv(arguments.output_dir / "training_building_quality.csv")
    summaries = []
    for building in buildings:
        output_dir = arguments.output_dir if len(buildings) == 1 else arguments.output_dir / building
        summaries.append(write_building(frame[[building]], building, output_dir, metadata, digest,
                                       metadata_digest, int(len(eligible)), arguments.building is not None))
    if len(buildings) == 1:
        summary = summaries[0]
    else:
        common_keys = ["dataset", "source", "raw_sha256", "metadata_sha256", "unit", "time_basis",
                       "selection_rule", "eligible_training_buildings", "processing"]
        summary = {key: summaries[0][key] for key in common_keys}
        summary.update({
            "num_buildings": len(buildings),
            "building_ids": buildings,
            "buildings": [{
                "building_id": item["building_id"],
                "data_dir": item["building_id"],
                "data_file": f"{item['building_id']}/building.csv",
                "summary_file": f"{item['building_id']}/data_summary.json",
                "processed_sha256": item["processed_sha256"],
            } for item in summaries],
        })
        (arguments.output_dir / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
