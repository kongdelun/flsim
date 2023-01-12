from pathlib import Path
from pandas import DataFrame, Series
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils.result import print_df


def parse_tb_scalar(path, scalar: str):
    steps, values = [], []
    ea = EventAccumulator(path)
    ea.Reload()
    sks = ea.scalars.Keys()
    if scalar not in sks:
        raise KeyError(scalar)
    for it in ea.scalars.Items(scalar):
        steps.append(it.step)
        values.append(it.value)
    return steps, values


def get_scalar(src: str, scalar: str, step: int = 1, excluded: list[str] = None):
    data = {}
    for d in Path(src).iterdir():
        if excluded and d.name in excluded:
            continue
        data[d.name] = parse_tb_scalar(d.absolute(), scalar)[1]
        for sd in filter(lambda x: x.is_dir(), d.iterdir()):
            tmp = parse_tb_scalar(sd.absolute(), scalar)[1]
            delta_n = (len(data[d.name]) - len(tmp))
            data[f'{d.name}/{sd.name}'] = tmp[:delta_n] if delta_n < 0 else [None] * delta_n + tmp
    df = DataFrame(data)
    df.index *= step
    return df


def get_acc_table(src: str, axis=0, verbose=False):
    tmp, data = {}, {}
    for d in filter(lambda x: x.is_dir() and not x.stem.startswith('__'), Path(src).iterdir()):
        for t in d.joinpath('output').iterdir():
            tmp[t.name] = [
                str(round(max(parse_tb_scalar(t.absolute(), 'test/{}'.format('acc'))[1]) * 100, 3))
            ]
            tmp[t.name].extend([
                str(round(max(parse_tb_scalar(st.absolute(), 'test/{}'.format('acc'))[1]) * 100, 3))
                for st in filter(lambda x: x.is_dir(), t.iterdir())
            ])
        data[d.name] = Series(data={k: '\t'.join(tmp[k]) for k in tmp})
        tmp.clear()
    df = DataFrame(data=data)
    df = DataFrame(df.values.T, index=df.columns, columns=df.index) if axis == 1 else df
    if verbose:
        print_df(df)
    return df

