import importlib

#
# logging.basicConfig(
#     stream=sys.stdout,
#     level=logging.INFO,
#     format='%(asctime)s %(levelname)s %(name)s: %(message)s'
# )
#
# # for i , g in zip(range(4),range(10)):
# #     print(i)
# import traceback
#
# args = {
#     'opt': {'lr': 0.1},
#     'batch_size': 32,
#     'epoch': 5
# }
#
# # for i in range(10):
# #     print(dict({'grad': i}, **args))
#
# try:
#     a = 1 / 0
# except:
#     print(traceback.format_exc())
#
#
# print('ok')
from benchmark.ctx import mnist

net, fds, cfg = mnist()
m = importlib.import_module('trainer.algorithm.fedavgm')
cls = getattr(m, 'FedAvgM')
trainer = cls(net, fds, **cfg)
trainer.start()
