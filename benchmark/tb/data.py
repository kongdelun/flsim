from pathlib import Path
from pandas import DataFrame
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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
