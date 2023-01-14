from pathlib import Path
from pandas import Series, DataFrame
from benchmark.tb.data import parse_tb_scalar
from utils.result import print_df


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
