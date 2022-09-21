import sys

from tqdm import trange, tqdm
from random import random, randint
import time

#
# for r in range(100):
#     with tqdm(total=3 if r % 5 == 0 else 2, desc="round %s" % r, colour='green', mininterval=1) as t:
#         print(t.bar_format)
# #         # 设置进度条右边显示的信息
# #         t.set_postfix(step='train', loss=random(), acc=randint(0, 100), )
# #         t.update(1)
# #         time.sleep(0.1)
# #         t.set_postfix(step='test', loss=random(), acc=randint(0, 100), )
# #         t.update(1)
# #         time.sleep(0.1)
# #
# #         if r % 5 == 0:
# #             t.set_postfix(step='test_all', loss=random(), acc=randint(0, 100), )
# #             time.sleep(0.1)
# #             t.update(1)
# #
# # for i in trange(4, desc='1st loop'):
# #     for j in trange(5, desc='2nd loop'):
# #         for k in trange(50, desc='3rd loop', leave=False):
# #             time.sleep(0.01)
# #

for i in tqdm(range(10), file=sys.stdout):
    tqdm.write('come on')
    time.sleep(0.1)
