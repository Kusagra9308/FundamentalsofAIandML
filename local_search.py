# local_search.py
"""
Local search replanning / path repair.
Two strategies implemented:
 - hill_climbing_random_restarts
 - simulated_annealing_repair

Both operate on path-space: they try to replace subpaths with alternative subpaths found by A* between intermediate nodes.
Requires search.py functions (astar) to be importable.
"""

import random
import math
from copy import deepcopy
import search

def path_cost(grid, path):
    if not path: return math.inf
    return sum(grid.cost(r,c) for (r,c) in path[1:])

def _shortest_between(grid, a, b):
    """A* between two nodes a->b using search.astar. Returns path or None."""
    res = search.astar(grid, a, b)
    return res['path'] if res else None

def random_subpath_indices(path, max_span=6):
    n = len(path)
    if n <= 3:
        return None
    i = random.randint(1, max(1, n-3))
    j = min(n-2, i + random.randint(0, min(max_span, n-3)))
    if i >= j:
        return None
    return i, j

def hill_climbing_random_restarts(grid, start, goal, restarts=10, iters_per_restart=200):
    best_path_res = search.astar(grid, start, goal)
    if not best_path_res:
        return None
    best_path = best_path_res['path']
    best_cost = path_cost(grid, best_path)

    for r in range(restarts):
        current_path = deepcopy(best_path)
        current_cost = best_cost
        for it in range(iters_per_restart):
            idxs = random_subpath_indices(current_path)
            if not idxs:
                continue
            i,j = idxs
            a = current_path[i-1]
            b = current_path[j+1]
            sub = _shortest_between(grid, a, b)
            if not sub:
                continue
            # new path: prefix up to i-1, sub (excluding endpoints to avoid duplicates), suffix from j+1
            new_path = current_path[:i] + sub[1:-1] + current_path[j+1:]
            new_cost = path_cost(grid, new_path)
            if new_cost < current_cost:
                current_path = new_path
                current_cost = new_cost
                if new_cost < best_cost:
                    best_path = new_path
                    best_cost = new_cost
        # random restart - perturb best as starting point for next restart
        # small random shuffle/pivot
        if len(best_path) > 4:
            a = random.randint(1, len(best_path)-3)
            b = min(len(best_path)-2, a + random.randint(1,3))
            # try replacing subpath a..b with A* between endpoints
            sub = _shortest_between(grid, best_path[a-1], best_path[b+1])
            if sub:
                candidate = best_path[:a] + sub[1:-1] + best_path[b+1:]
                c_cost = path_cost(grid, candidate)
                if c_cost < best_cost:
                    best_path = candidate
                    best_cost = c_cost
    return {'path': best_path, 'cost': best_cost}

def simulated_annealing_repair(grid, start, goal, initial_temp=5.0, cooling_rate=0.995, iterations=2000):
    base_res = search.astar(grid, start, goal)
    if not base_res:
        return None
    current_path = list(base_res['path'])
    current_cost = path_cost(grid, current_path)
    best_path = list(current_path)
    best_cost = current_cost
    T = initial_temp

    for it in range(iterations):
        idxs = random_subpath_indices(current_path)
        if not idxs:
            T *= cooling_rate
            continue
        i,j = idxs
        a = current_path[i-1]
        b = current_path[j+1]
        sub = _shortest_between(grid, a, b)
        if not sub:
            T *= cooling_rate
            continue
        new_path = current_path[:i] + sub[1:-1] + current_path[j+1:]
        new_cost = path_cost(grid, new_path)
        delta = new_cost - current_cost
        accept = False
        if delta <= 0:
            accept = True
        else:
            prob = math.exp(-delta / max(1e-9, T))
            if random.random() < prob:
                accept = True
        if accept:
            current_path = new_path
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_path = new_path
        T *= cooling_rate
    return {'path': best_path, 'cost': best_cost}
