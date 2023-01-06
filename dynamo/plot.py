import matplotlib.pyplot as plt
import pandas as pd
from compare import Compare
from matplotlib.patches import Patch
from runner import _unpick


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


# Load the results: will be a list of torch Measurement
results = _unpick('output-benchmark-20230106_173208.pickle')

# Transform into a table
compare = Compare(results)
df = compare.join_to_df()

# Preprocess the df
df = preprocess(df)


# Cast the column types
df['has_warnings'] = df['has_warnings'].replace({'False': False, 'True': True})
df = df.convert_dtypes()

# Get the module name and the operation itself
df['op_name'] = df['op'].apply(lambda x: x.split('.')[-1])
df['module'] = df['op'].apply(lambda x: '.'.join(x.split('.')[:-1]))

# generate a df for each operation
df_by_op = df.groupby('op_name')


def plot(df, name=''):
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
        'kornia_gpu': {
            'idx': 3, 'new_name': 'eager_gpu', 'color': (.698, .878, .38, .1),
        },
        'dynamo_kornia_gpu': {
            'idx': 4,  'new_name': 'dynamo_gpu',
            'color': (.741, .494, .745, 1.),
        },
    }

    cols = [x for x in df.columns if x not in _map.keys()]

    # Join time into a unique serie
    _df = pd.DataFrame(columns=cols+['optimizer', 'time', 'old_name'])
    for k, v in _map.items():
        if k not in df.columns:
            continue
        df_tmp = df[cols + [k]]
        df_tmp = df_tmp.rename({k: 'time'}, axis='columns')
        df_tmp['optimizer'] = v['new_name']
        df_tmp['old_name'] = k
        _df = pd.concat([_df, df_tmp])

    _df['time'] = _df['time'].astype(float)
    time_unit = _df['time_unit'].unique()[0]

    _df['label'] = _df['threads'] + ' threads | ' + _df['arguments']
    _df['_seq'] = _df['old_name'].apply(lambda x: _map[x]['idx'])
    _df['color'] = _df['old_name'].apply(lambda x: _map[x]['color'])
    _df = _df.sort_values(['label', '_seq'])

    # Create graph
    ax = _df.plot.bar(
        x='label',
        y='time',
        color=_df['color'].values.tolist(),
        legend=True,
        title=name,
        ylabel=time_unit,
        xlabel='arguments',
        logy=True,
    )

    # Create legend
    handles = []
    _as = []
    for _, row in _df.iterrows():
        if row['optimizer'] in _as:
            continue
        handles.append(Patch(facecolor=row['color'], label=row['optimizer']))
        _as.append(row['optimizer'])
    ax.legend(handles=handles)

    # Add labels to bars
    ax.bar_label(ax.containers[0])

    # Config plot
    plt.xticks(rotation=45)

    # show or save
    plt.show()


# Plot each operation
for op_name, frame in df_by_op:
    plot(frame, op_name)
