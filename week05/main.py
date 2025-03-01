from enum import Enum
from functools import lru_cache
from itertools import product
from pprint import pp
from time import sleep, perf_counter
from types import FunctionType


def delItem(tup, index):
    return tup[:index] + tup[index + 1 :]


def setItem(tup, index, value):
    return tup[:index] + (value,) + tup[index + 1 :]


def takeN(piles, take, index):
    if piles[index] - take == 0:
        return delItem(piles, index)
    return setItem(piles, index, piles[index] - take)


@lru_cache(None)
def successors(piles):
    nextPiles = []
    for index in range(len(piles)):
        pile = piles[index]
        for take in range(1, min(pile + 1, 4)):
            successor = takeN(piles, take, index)
            nextPiles.append(successor)
    return nextPiles


def maxChoice(piles):
    maxEval = float("-inf")
    succs = successors(tuple(piles))
    # print(succs)
    for s in succs:
        eval = minimax_value((s, 2))
        maxEval = max(maxEval, eval)

    return maxEval


def minChoice(piles):
    minEval = float("inf")
    succs = successors(tuple(piles))
    # print(succs)
    for s in succs:
        eval = minimax_value((s, 1))
        minEval = min(minEval, eval)
    return minEval


def minimax_value(state):
    piles = state[0]
    turn = state[1]
    if len(piles) == 0:
        return -1 if turn == 2 else 1

    if turn == 1:
        return maxChoice(piles)
    else:
        return minChoice(piles)


def measurePerf(fn: FunctionType, *args):
    t1_start = perf_counter()
    res = fn(*args)
    t1_stop = perf_counter()
    print("Time measured: ", t1_stop - t1_start)
    return res


# print(minimax_value(State([2, 1], Turn.MAX)))
