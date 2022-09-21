from pathlib import Path

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

root = 'D:/project/python/MyFed/benchmark/{}/{}/output/{}'


def fix_tb(src: str, step = 5):
    src = Path(src)
    dn, dp = src.parts[-2], src.parts[-1]
    for f in src.iterdir():
        fp = f.name.split('-')
        data = pd.read_csv(f)
        with SummaryWriter(root.format(dn, dp, fp[1].replace('_', '/'))) as w:
            for s, v in zip(data['Step'].tolist(), data['Value'].tolist()):
                if s % step == 0:
                    w.add_scalar('test_all/{}'.format('acc' if 'acc' in fp[-1] else 'loss'),
                                 scalar_value=v, global_step=s)


if __name__ == '__main__':
    fix_tb('./src/cifar10/iid')
