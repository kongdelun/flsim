import mysys
from dataclasses import dataclass
import pyfiglet
from tqdm import tqdm
from pathlib import Path
from termcolor import cprint
from typing import Optional
from pandas import DataFrame, Series, option_context

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class Style:
    color: Optional[str]
    on_color: Optional[str]
    attrs: Optional[list]


Styles = [
    Style('yellow', None, ['reverse']),
    Style('green', None, ['reverse']),
    Style('red', None, ['reverse']),
    Style('blue', None, ['reverse']),
    Style('cyan', 'on_grey', None),
    Style('red', None, ['underline'])
]


def progress_bar(total_step, desc=None):
    return tqdm(
        total=total_step,
        desc=desc,
        colour='green',
        bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]',
        file=sys.stdout,
    )


def print_text(text, style: int):
    style = Styles[style]
    cprint(text, style.color, style.on_color, style.attrs)


def print_banner(text):
    f = pyfiglet.Figlet(font="slant")
    print(f.renderText(text))


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
                str(round(max(parse_tb_scalar(t.absolute(), 'test_all/{}'.format('acc'))[1]) * 100, 3))
            ]
            tmp[t.name].extend([
                str(round(max(parse_tb_scalar(st.absolute(), 'test_all/{}'.format('acc'))[1]) * 100, 3))
                for st in filter(lambda x: x.is_dir(), t.iterdir())
            ])
        data[d.name] = Series(data={k: '\t'.join(tmp[k]) for k in tmp})
        tmp.clear()
    df = DataFrame(data=data)
    df = DataFrame(df.values.T, index=df.columns, columns=df.index) if axis == 1 else df
    if verbose:
        with option_context('expand_frame_repr', False, 'display.max_rows', None):
            print(df)
    return df
