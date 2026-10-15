import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot all test days and the first seven complete forecast days.")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/first_run"))
    arguments = parser.parse_args()
    predictions = pd.read_csv(arguments.run_dir / "predictions.csv", parse_dates=["timestamp", "origin"])
    scores = pd.read_csv(arguments.run_dir / "metrics.csv")
    configuration = json.loads((arguments.run_dir / "config.json").read_text())
    test = predictions[predictions["split"] == "test"].copy()
    selected_origins = test["origin"].drop_duplicates().head(7)
    example = test[test["origin"].isin(selected_origins)].set_index("timestamp").asfreq("h")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained", gridspec_kw={"height_ratios": [2, 1]})
    styles = [
        ("actual_kwh", "Observed", "#242424", "-"),
        ("weekly_naive_kwh", "Same hour last week", "#8B8B8B", "--"),
        ("transformer_kwh", "Transformer", "#2374AB", "-"),
    ]
    for column, label, color, line_style in styles:
        axes[0].plot(example.index, example[column], label=label, color=color, linestyle=line_style, linewidth=1.4)
    axes[0].set_title(f"{configuration['data']['building_id']} | {test['timestamp'].dt.year.iloc[0]}, first 7 test forecast days", loc="left")
    axes[0].set_ylabel("Hourly electricity (kWh)")
    axes[0].legend(ncol=3, fontsize=9, loc="upper left")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    daily = pd.DataFrame({"origin": test["origin"]})
    for column, label, color in [("weekly_naive_kwh", "Same hour last week", "#8B8B8B"), ("transformer_kwh", "Transformer", "#2374AB")]:
        daily[label] = (test[column] - test["actual_kwh"]).abs()
        values = daily.groupby("origin")[label].mean()
        axes[1].plot(values.index, values, color=color, label=label, linewidth=1)
    axes[1].set_title("Daily mean absolute error | all retained test days", loc="left")
    axes[1].set_ylabel("MAE (kWh)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for axis in axes:
        axis.grid(axis="y", alpha=0.15)
    figure.savefig(arguments.run_dir / "forecast_comparison.png", dpi=180)
    figure.savefig(arguments.run_dir / "forecast_comparison.svg")
    plt.close(figure)
    test_scores = scores[scores["split"] == "test"]
    print(test_scores.to_string(index=False))
    print(f"Saved {arguments.run_dir / 'forecast_comparison.png'}")


if __name__ == "__main__":
    main()
