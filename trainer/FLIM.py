import dataclasses
import random
from typing import Sequence

from utils.tool import set_seed


@dataclasses.dataclass
class Bid:
    cid: int
    num_sample: int
    price: float
    pay: float = 0.


def internal_auction(budget: float, bids: Sequence[Bid]):
    sorted_bids = sorted(bids, key=lambda b: b.price / b.num_sample)
    for i in range(len(sorted_bids)):
        if budget / sum([b.num_sample for b in sorted_bids[:i + 1]]) < sorted_bids[i].price / sorted_bids[i].num_sample:
            for j in range(i):
                sorted_bids[j].pay = sorted_bids[j].num_sample * min(
                    budget / sum([b.num_sample for b in sorted_bids[:i]]),
                    sorted_bids[i].price / sorted_bids[i].num_sample
                )
            break
    return [b.cid for b in bids if b.pay > 0.]


set_seed(2077)

fbids = [Bid(i, random.randint(100, 500), random.uniform(2, 4)) for i in range(100)]
print(fbids)
print(internal_auction(100, fbids))
