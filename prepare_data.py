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


def main():
    parser = argparse.ArgumentParser(description="Prepare one BDG2 building without using test data for selection.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--building", help="Explicit building ID; omit for training-only quality selection.")
    parser.add_argument("--download", action="store_true")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
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
    )
    quality.index.name = "building_id"
    quality = quality.sort_values(["missing_fraction", "nonpositive_fraction", "building_id"])
    quality.to_csv(arguments.output_dir / "training_building_quality.csv")
    eligible = quality[quality["eligible"]]
    if arguments.building is None and eligible.empty:
        raise ValueError("No building passes the predefined training-data quality rule.")
    building = arguments.building or eligible.index[0]
    frame = pd.read_csv(electricity_path, usecols=["timestamp", building], parse_dates=["timestamp"])
    frame = frame.set_index("timestamp").rename(columns={building: "load_kwh"})
    expected_index = pd.date_range("2016-01-01", "2017-12-31 23:00", freq="h")
    if not frame.index.equals(expected_index):
        raise ValueError("Unexpected duplicate, missing, or unordered timestamps in the raw data.")
    invalid = ~np.isfinite(frame["load_kwh"]) | (frame["load_kwh"] < 0)
    frame.loc[invalid, "load_kwh"] = np.nan
    frame["split"] = np.where(frame.index < TRAIN_END, "train", np.where(frame.index < VALIDATION_END, "validation", "test"))
    frame.to_csv(arguments.output_dir / "building.csv")
    frame["load_kwh"].resample("MS").agg(["count", "mean", "std", "min", "max"]).to_csv(arguments.output_dir / "monthly_load_summary.csv")
    metadata = pd.read_csv(metadata_path)
    matching = metadata.loc[metadata["building_id"] == building]
    summary = {
        "dataset": "Building Data Genome 2 (raw electricity)",
        "source": f"{SOURCE_ROOT}/meters/raw/electricity.csv",
        "raw_sha256": digest,
        "metadata_sha256": file_hash(metadata_path),
        "processed_sha256": file_hash(arguments.output_dir / "building.csv"),
        "building_id": building,
        "building_metadata": json.loads(matching.to_json(orient="records")),
        "unit": "kWh per hourly interval",
        "time_basis": "Source local-time hourly labels, retained without timezone conversion",
        "selection_rule": "Explicit ID" if arguments.building else "Education/office; 2016 missing <=0.5%, nonpositive <=1%, positive mean/std; sort missing, nonpositive, ID",
        "eligible_training_buildings": int(len(eligible)),
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
    (arguments.output_dir / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
