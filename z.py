from bitsets import bitset
import numpy as np
import itertools
import random

if __name__ == "__main__":
    x = np.arange(16)
    # y = np.array([1, 4, 5])
    # print(set(x).intersection(set(y)))
    X = itertools.chain.from_iterable(itertools.combinations(x, y) for y in range(len(x) + 1))
    X = list(X)
    random.shuffle(X)
    print(len(X))
