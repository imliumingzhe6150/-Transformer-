"""Plot actual hourly loads, weekly baseline and one fitted Transformer per building."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('outputs/.matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/multi_building_2026-10-23'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', choices=['shared', 'individual'], default='shared')
    args = parser.parse_args()
    folder = args.run_dir / f'seed_{args.seed}'
    output = args.run_dir / 'figures' / f'{args.model}_seed_{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.run_dir / 'experiment_plan.json').read_text())
    buildings = plan['cohort']['building_ids']
    font = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=font).get_name()
    else:
        family = 'DejaVu Sans'
    plt.rcParams.update({'font.family': family, 'font.size': 11, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    label = '共享 Transformer' if args.model == 'shared' else '单独训练 Transformer'
    styles = [('actual_kwh', '真实用电量', '#252525', '-'),
              ('weekly_naive_kwh', '上周同期基线', '#D08A36', '--'),
              ('prediction_kwh', label, '#2879AC', '-')]
    shared = pd.read_csv(folder / 'shared/predictions.csv', parse_dates=['timestamp', 'origin']) if args.model == 'shared' else None
    figure, axes = plt.subplots(len(buildings), 2, figsize=(16, 14), squeeze=False)
    full, full_axes = plt.subplots(len(buildings), 1, figsize=(16, 14), squeeze=False)
    records, score_rows, all_data = [], [], []
    for i, building in enumerate(buildings):
        single = pd.read_csv(folder / 'individual' / building / 'predictions.csv', parse_dates=['timestamp', 'origin'])
        frame = single[single.split == 'test'].copy()
        if shared is not None:
            other = shared[(shared.building_id == building) & (shared.split == 'test')]
            frame = frame.merge(other, on=['split', 'origin', 'timestamp', 'horizon_hour'],
                                validate='one_to_one', how='outer', suffixes=('', '_shared'), indicator=True)
            assert frame['_merge'].eq('both').all()
            assert np.array_equal(frame.actual_kwh, frame.actual_kwh_shared)
            frame['prediction_kwh'] = frame.shared_transformer_kwh
        else:
            frame['prediction_kwh'] = frame.transformer_kwh
        frame = frame.sort_values('timestamp')
        assert not frame.timestamp.duplicated().any()
        assert np.isfinite(frame[['actual_kwh', 'weekly_naive_kwh', 'prediction_kwh']]).all().all()
        origins = frame.origin.drop_duplicates().iloc[:7]
        example = frame[frame.origin.isin(origins)].set_index('timestamp').asfreq('h')
        entire = frame.set_index('timestamp').reindex(pd.date_range('2017-07-01', '2017-12-31 23:00', freq='h'))
        short = building.split('_')[-1]
        for column, name, color, style in styles:
            axes[i, 0].plot(example.index, example[column], color=color, ls=style, lw=1.25, label=name)
            full_axes[i, 0].plot(entire.index, entire[column], color=color, ls=style, lw=.75, alpha=.85, label=name)
        metrics = {}
        for column, name, color, style in styles[1:]:
            error = frame[column] - frame.actual_kwh
            daily = pd.Series(error.abs().to_numpy(), index=frame.origin).groupby(level=0).mean()
            daily = daily.reindex(pd.date_range('2017-07-01', '2017-12-31', freq='D'))
            axes[i, 1].plot(daily.index, daily, color=color, ls=style, lw=1.1, label=name)
            metrics[column] = (float(error.abs().mean()), float(np.sqrt(np.mean(error**2))))
            score_rows.append(dict(building_id=building, method=name, seed=args.seed,
                                   test_hours=len(frame), mae_kwh=metrics[column][0], rmse_kwh=metrics[column][1]))
        base, model = metrics['weekly_naive_kwh'], metrics['prediction_kwh']
        axes[i, 0].set_title(f'{chr(97+i)}  {short}｜最初 7 个有效预测日', loc='left', fontsize=12)
        axes[i, 1].set_title(f'全测试期每日 MAE｜基线 {base[0]:.2f} → 模型 {model[0]:.2f} kWh', loc='left', fontsize=11)
        full_axes[i, 0].set_title(f'{short}｜全期 RMSE：基线 {base[1]:.2f} → 模型 {model[1]:.2f} kWh', loc='left', fontsize=12)
        for ax in [axes[i, 0], full_axes[i, 0]]:
            ax.set_ylabel('每小时电量 (kWh)')
            ax.set_ylim(bottom=0)
        axes[i, 1].set_ylabel('MAE (kWh)')
        axes[i, 1].set_ylim(bottom=0)
        for ax in [axes[i, 0], axes[i, 1], full_axes[i, 0]]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
            ax.grid(axis='y', alpha=.15)
            ax.margins(x=.01)
        records.append(dict(building_id=building, test_hours=len(frame), detail_hours=len(example),
                            detail_start=str(example.index.min()), detail_end=str(example.index.max())))
        all_data.append(frame[['timestamp', 'origin', 'actual_kwh', 'weekly_naive_kwh', 'prediction_kwh']].assign(building_id=building))
    for fig, axlist, title in [(figure, axes, '五栋建筑：预测细节与全测试期误差'),
                               (full, full_axes, '五栋建筑：全测试期逐小时预测')]:
        fig.suptitle(f'{title}｜{label}｜种子 {args.seed}', fontsize=19, y=.985)
        handles, labels = axlist[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .955), ncol=3, fontsize=12)
        fig.subplots_adjust(top=.90, bottom=.065, left=.075, right=.98, hspace=.55, wspace=.22)
        fig.text(.075, .018, '日期为原数据当地时间标签（2017 年）；各建筑纵轴范围不同。缺失窗口留白，不插值。\n'
                 '曲线来自一个固定种子的实际模型；不是三个种子预测值的平均。误差覆盖全部有效测试小时。', fontsize=10, color='#555555')
    for fig, name in [(figure, 'forecast_detail_and_error'), (full, 'forecast_full_test')]:
        fig.savefig(output / f'{name}.png', dpi=300)
        fig.savefig(output / f'{name}.svg')
        fig.savefig(output / f'{name}.pdf')
        plt.close(fig)
    pd.DataFrame(score_rows).to_csv(output / 'displayed_metrics.csv', index=False)
    pd.concat(all_data).to_csv(output / 'plot_source_data.csv', index=False)
    (output / 'figure_notes.json').write_text(json.dumps(dict(
        purpose='Compare observed load, weekly persistence and Transformer; show timing and magnitude of errors without selecting favorable dates.',
        archetype='quantitative grid', backend='python', seed=args.seed, model=args.model,
        exports='16 x 14 inches for screen/report, PNG 300 dpi and editable SVG/PDF; no journal submission claim',
        detail_rule='First seven retained test forecast days per building; illustrative detail only, not a substitute for full-period curves.',
        full_period_rule='All retained hourly test observations; reindex missing test hours as gaps; no interpolation or smoothing.',
        daily_metric='Mean absolute error over each retained 24-hour forecast; displayed overall MAE/RMSE computed on all retained test hours.',
        uncertainty='No uncertainty bands; a single fixed seed, no ensemble or significance inference.',
        mapping='actual_kwh=observed; weekly_naive_kwh=lag 168 h; prediction_kwh=selected model output, all kWh per hourly interval',
        rows=records), ensure_ascii=False, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
