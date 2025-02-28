from functools import lru_cache
from itertools import product
from pprint import pp
from time import sleep, perf_counter
from types import FunctionType


def expandRoot(pile: int):
    path = []
    for subtract in range(1, min(4, pile + 1)):  # end is exclusive
        path.append(pile - subtract)
    return tuple(path)


def stripEmpty(piles: list[tuple]) -> list[tuple]:
    it = filter(lambda x: any(y != 0 for y in x), piles)
    return list(it)


def sortBy():
    pass


def prettyPrint(paths: list[tuple], width=100):
    pp(paths, width=width, compact=True)


def measurePerf(fn: FunctionType, args):
    t1_start = perf_counter()
    res = fn(*args)
    t1_stop = perf_counter()
    print("Time measured: ", t1_stop - t1_start)
    return res


def processSubPaths(paths: list[tuple]):
    foundPaths = set()
    for pile in paths:
        # print("pile: ", pile)
        expandedPaths = set(sorted(customProductPaths(pile, expandPaths(pile))))
        # print("expandedPaths: ", expandedPaths)
        if len(expandedPaths) > 0:
            foundPaths = foundPaths.union(expandedPaths)
    # print("foundPaths: ", foundPaths)
    compiledPaths = []
    for path in foundPaths:
        if len(path) > 0:
            subpaths = processSubPaths([path])
            compiledPaths.append(subpaths)
    # prettyPrint(compiledPaths)
    if compiledPaths == []:
        return paths
    return [*paths, *compiledPaths]


def productPaths(paths: list[tuple]):
    combos = product(*paths)
    unique_Combos = set(tuple(sorted(combo)) for combo in combos)
    return unique_Combos


def take(piles, p, amount):
    # Allows only legal operations, i.e NOT take(3) from [1,0]
    if piles[p] >= amount:
        newPile = piles.copy()
        newPile[p] -= amount
        return [x for x in newPile if x > 0]
    return None


def Successors(state):
    currState = state[0]
    succ = set()
    for i in range(len(currState)):
        for j in range(1, 4):
            new_state = take(currState, i, j)
            if new_state is not None:
                succ.add(tuple(new_state))
    return [list(t) for t in succ]


def customProductPaths(piles: list[int]):
    combos = []
    paths = expandPaths(piles)
    for pathsIndex in range(0, len(paths)):
        for index in range(0, len(paths[pathsIndex])):
            combo = tuple(
                sorted(
                    [
                        paths[pathsIndex][index],
                        *piles[pathsIndex + 1 :],
                        *piles[:pathsIndex],
                    ]
                )
            )
            combos.append(combo)
    return combos


def expandPaths(piles: list[int]):
    # This respects the turn inside of State
    paths: list[tuple] = []
    for pile in piles:
        paths.append(tuple(expandRoot(pile)))
    return paths


# prettyPrint(processSubPaths([(0, 1), (1, 1), (0, 2)]))
