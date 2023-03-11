import os

import matplotlib.pyplot as plt
import pandas as pd
from compare import Compare

plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 10.0


def _color_op_names(df):
    def _update_name(row):
        if row['op'] == 'color.multiple':
            args = row['arguments'][1:-1].split(',')
            row['op'] = 'color.' + args[-1].lstrip()
            row['arguments'] = '[' + ','.join(args[:-1]) + ']'
        return row

    return df.apply(lambda row: _update_name(row), axis=1)


def preprocess(df):
    # add here the call of the desired function to do something with the df
    # e.g. fix the color operation because we use a multiple case
    df = _color_op_names(df)

    df = df.replace(r'^\s*$', None, regex=True)

    return df


def _format_time(value: float | str, unit: str) -> str:
    if unit == 'us':
        if value > 1e5:
            value = value / 1e6
            unit = 's'
        elif value > 1e2:
            value = value/1e3
            unit = 'ms'

    return f'{value:.2f} {unit}'


def _show_or_save(
    save: bool = False, outdir: str = './', name: str = 'graph.png',
) -> None:
    if save:
        os.makedirs(outdir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, name),
            dpi=600,
            bbox_inches='tight',
        )
        return

    plt.show()


def _plot_time_graph(
    df: pd.DataFrame,
    colors: list[tuple[float, ...]],
    name: str = 'My graph',
    time_unit: str = '',
) -> plt.Axes:
    ax = df.plot.barh(
        color=colors,
        legend=True,
        title=name,
        width=0.8,
        align='center',
        logx=True,
    )
    ax.invert_yaxis()
    ax.margins(0.1)

    # Config legend
    ax.legend(fontsize='xx-small')

    # Add labels to bars
    for container in ax.containers:
        labels = [
            _format_time(float(v), time_unit)
            for v in container.datavalues
        ]
        ax.bar_label(container, labels=labels, fontsize=5, padding=6)

    # Config plot
    plt.xlabel(f'Time ({time_unit})', fontsize='x-small')
    plt.ylabel('Argunments', fontsize='x-small')
    plt.xticks(rotation=45, fontsize='xx-small')
    plt.yticks(fontsize='xx-small')
    plt.tick_params(axis='y', which='major')

    return ax


def _plot_ratio_graph(
    df: pd.DataFrame,
    colors: list[tuple[float, ...]],
    name: str = 'My ratio graph',
) -> plt.Axes:
    ax = df.plot.barh(
        color=colors,
        legend=True,
        width=0.8,
        align='center',
    )
    ax.invert_yaxis()
    ax.margins(0.1)

    # Config legend
    ax.legend(fontsize='xx-small')

    # Add labels to bars
    for container in ax.containers:
        labels = [
            f'{(float(v)):.3f}x' if float(v) < 1 else f'{(float(v)):.1f}x'
            for v in container.datavalues
        ]
        ax.bar_label(container, labels=labels, fontsize=5, padding=6)

    # Config plot
    plt.ylabel('Argunments', fontsize='x-small')
    plt.xticks(rotation=45, fontsize='xx-small')
    plt.yticks(fontsize='xx-small')
    plt.title(name)
    plt.tick_params(axis='y', which='major')

    return ax


def plot(df, name='', save: bool = False, outdir: str = 'out_graphs/'):
    print(f'Working on plot of {name}...')

    if df['has_warnings'].any():
        print(
            '\033[1;33m'
            '(! XX%) Measurement has high variance, where XX is the IQR'
            ' / median * 100.'
            '\033[0;0m',
        )

    _map = {
        'opencv_cpu': {
            'idx': 0, 'new_name': 'opencv', 'color': (.627, .627, .627, 1.),
        },
        'kornia_cpu': {
            'idx': 1, 'new_name': 'eager_cpu', 'color': (.745, .725, .859, 1.),
        },
        'dynamo_kornia_cpu': {
            'idx': 2, 'new_name': 'dynamo_cpu', 'color': (.99, .8, .898, 1.),
        },
        'kornia_cuda': {
            'idx': 3, 'new_name': 'eager_gpu', 'color': (.698, .878, .38, 1.),
        },
        'dynamo_kornia_cuda': {
            'idx': 4,  'new_name': 'dynamo_gpu',
            'color': (.741, .494, .745, 1.),
        },
    }

    cols = [x for x in df.columns if x not in _map.keys()]

    # Join time into a unique serie
    _df = pd.DataFrame(columns=cols+['time', 'optimizer'])
    for k, v in _map.items():
        if k not in df.columns:
            continue
        df_tmp = df[cols + [k]]
        df_tmp = df_tmp.rename({k: 'time'}, axis='columns')
        df_tmp['optimizer'] = k
        _df = pd.concat([_df, df_tmp])

    _df['time'] = _df['time'].astype(float)
    time_unit = _df['time_unit'].unique()[0]

    _df['label'] = _df['threads'] + ' threads\n' + _df['arguments']

    _df['label'] = _df['label'].astype('category')

    df_pivot = pd.pivot_table(
        _df,
        values='time',
        index='label',
        columns='optimizer',
        fill_value=0.,
    )

    df_pivot = df_pivot.sort_values(by=['label'], ascending=False)

    cols = df_pivot.columns.tolist()
    cols.sort(key=lambda x: _map[x]['idx'])
    df_pivot = df_pivot.reindex(cols, axis='columns')
    df_pivot = df_pivot.rename(columns={c: _map[c]['new_name'] for c in cols})

    print(df_pivot.to_string())

    colors = [m['color'] for m in _map.values()]

    # Time graph
    _ = _plot_time_graph(df_pivot, colors, name, time_unit)
    _show_or_save(save, outdir, f'{name}.png')

    # Ratio by opencv
    _n = df_pivot['opencv']
    df_ratio_cv2 = df_pivot.drop(columns=['opencv'])
    for c in df_ratio_cv2.columns:
        df_ratio_cv2[c] = _n / df_ratio_cv2[c]

    _ = _plot_ratio_graph(df_ratio_cv2, colors[1:], name)
    plt.xlabel('times faster than OpenCV', fontsize='x-small')
    _show_or_save(save, outdir, f'{name}_ratio.png')

    # Ratio between cpu ops
    _n = df_pivot['eager_cpu']
    df_ratio_cpu = df_pivot.drop(
        columns=[
            'eager_cpu', 'eager_gpu', 'dynamo_gpu',
        ],
    )
    for c in df_ratio_cpu.columns:
        df_ratio_cpu[c] = _n / df_ratio_cpu[c]

    _ = _plot_ratio_graph(df_ratio_cpu, colors[1:], name)
    plt.xlabel('times faster than eager Kornia on CPU', fontsize='x-small')
    _show_or_save(save, outdir, f'{name}_ratio_cpu.png')

    # Ratio between gpu ops
    _n = df_pivot['eager_gpu']
    df_ratio_gpu = df_pivot.drop(
        columns=[
            'eager_gpu', 'eager_cpu', 'dynamo_cpu',
        ],
    )
    for c in df_ratio_gpu.columns:
        df_ratio_gpu[c] = _n / df_ratio_gpu[c]

    _ = _plot_ratio_graph(df_ratio_gpu, colors[-2:], name)
    plt.xlabel('times faster than eager Kornia on GPU', fontsize='x-small')
    _show_or_save(save, outdir, f'{name}_ratio_gpu.png')


def graphs_from_results(results) -> None:

    # Transform into a table
    compare = Compare(results)
    df = compare.join_to_df()

    # Preprocess the df
    df = preprocess(df)

    # Cast the column types
    df['has_warnings'] = df['has_warnings'].replace(
        {'False': False, 'True': True},
    )
    df = df.convert_dtypes()

    # Get the module name and the operation itself
    df['op_name'] = df['op'].apply(lambda x: x.split('.')[-1])
    df['module'] = df['op'].apply(lambda x: '.'.join(x.split('.')[:-1]))

    # generate a df for each operation
    df_by_op = df.groupby('op_name')

    # Plot each operation
    for op_name, frame in df_by_op:
        plot(frame, op_name, True)
