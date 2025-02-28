from enum import Enum
from functools import lru_cache
from itertools import product
from pprint import pp
from time import sleep, perf_counter
from types import FunctionType


class Turn(Enum):
    MAX = 2
    MIN = 1
    END = 0


def takeN(piles, take, index):
    newPile = piles.copy()
    if piles[index] - take == 0:
        del newPile[index]
        return sorted(newPile)
    newPile[index] -= take
    return sorted(newPile)


def successors(piles: list[int]):
    nextPiles = []
    for index in range(len(piles)):
        pile = piles[index]
        for take in range(1, pile + 1):
            successor = takeN(piles, take, index)
            if successor is not None and successor not in nextPiles:
                nextPiles.append(successor)
    return nextPiles


def maxChoice(piles):
    maxEval = float("-inf")
    for s in successors(piles):
        eval = minimax_value((s, 1))
        maxEval = max(maxEval, eval)

    return maxEval


def minChoice(piles):
    minEval = float("inf")
    for s in successors(piles):
        eval = minimax_value((s, 2))
        minEval = min(minEval, eval)
    return minEval


def measurePerf(fn: FunctionType, *args):
    t1_start = perf_counter()
    res = fn(*args)
    t1_stop = perf_counter()
    print("Time measured: ", t1_stop - t1_start)
    return res


def minimax_value(state):
    piles = state[0]
    turn = state[1]
    if len(piles) == 0:
        return -1 if turn == 2 else 1

    if turn == 1:
        return maxChoice(piles)
    elif turn == 2:
        return minChoice(piles)
    else:
        return 0


# print(minimax_value(State([2, 1], Turn.MAX)))
