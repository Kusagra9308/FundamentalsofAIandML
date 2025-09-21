# search.py
"""
Core grid, parsing, and search algorithms.
Provides: Grid, parse_map_file, load_schedule, bfs, ucs, astar, reconstruct_path, simulate_with_dynamic_replanning
"""

from collections import deque, defaultdict
import heapq
import time

class Grid:
    def __init__(self, rows:int, cols:int, cells:list):
        self.R = rows
        self.C = cols
        self.cells = cells  # 2D list: int cost >=1 or None for blocked

    def in_bounds(self, r, c):
        return 0 <= r < self.R and 0 <= c < self.C

    def passable(self, r, c):
        return self.in_bounds(r, c) and (self.cells[r][c] is not None)

    def cost(self, r, c):
        v = self.cells[r][c]
        if v is None:
            raise ValueError("Accessing cost of blocked cell")
        return v

    def neighbors4(self, r, c):
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if self.passable(nr, nc):
                yield (nr, nc)

def parse_map_file(path:str) -> Grid:
    with open(path) as f:
        header = f.readline().strip().split()
        if not header:
            raise ValueError("Empty map file")
        R, C = map(int, header[:2])
        cells = []
        for i in range(R):
            tokens = f.readline().strip().split()
            if len(tokens) < C:
                raise ValueError(f"Row {i} expects {C} tokens but got {len(tokens)}")
            row = []
            for tok in tokens[:C]:
                if tok.upper() == 'X':
                    row.append(None)
                else:
                    v = int(tok)
                    if v < 1:
                        raise ValueError("Cell cost must be >= 1")
                    row.append(v)
            cells.append(row)
    return Grid(R, C, cells)

def load_schedule(path:str):
    """
    schedule file lines: t r c
    returns: dict time -> set((r,c), ...)
    """
    sched = defaultdict(set)
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) < 3: continue
            t = int(parts[0]); r = int(parts[1]); c = int(parts[2])
            sched[t].add((r,c))
    return sched

# -------------------------
# utilities
# -------------------------
def reconstruct_path(came_from:dict, goal:tuple):
    path = []
    node = goal
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(node)
    path.reverse()
    return path

# -------------------------
# BFS / UCS / A*
# -------------------------
def bfs(grid:Grid, start:tuple, goal:tuple):
    """
    BFS (fewest steps). NOTE: ignores varying costs; useful for unit-cost environments.
    Returns dict: {path, cost, expanded, time}
    """
    t0 = time.perf_counter()
    frontier = deque([start])
    came_from = {}
    visited = {start}
    nodes_expanded = 0
    while frontier:
        curr = frontier.popleft()
        nodes_expanded += 1
        if curr == goal:
            elapsed = time.perf_counter() - t0
            path = reconstruct_path(came_from, goal)
            path_cost = sum(grid.cost(r,c) for (r,c) in path[1:])
            return {'path': path, 'cost': path_cost, 'expanded': nodes_expanded, 'time': elapsed}
        for nbr in grid.neighbors4(*curr):
            if nbr not in visited:
                visited.add(nbr)
                came_from[nbr] = curr
                frontier.append(nbr)
    return None

def ucs(grid:Grid, start:tuple, goal:tuple):
    """
    Uniform-cost search (Dijkstra). Minimizes path cost with varying cell costs.
    """
    t0 = time.perf_counter()
    pq = []
    heapq.heappush(pq, (0, start))
    came_from = {}
    cost_so_far = {start: 0}
    nodes_expanded = 0
    while pq:
        g, curr = heapq.heappop(pq)
        nodes_expanded += 1
        if curr == goal:
            elapsed = time.perf_counter() - t0
            path = reconstruct_path(came_from, goal)
            return {'path': path, 'cost': g, 'expanded': nodes_expanded, 'time': elapsed}
        for nbr in grid.neighbors4(*curr):
            nr, nc = nbr
            move_cost = grid.cost(nr, nc)
            new_cost = g + move_cost
            if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                cost_so_far[nbr] = new_cost
                came_from[nbr] = curr
                heapq.heappush(pq, (new_cost, nbr))
    return None

def astar(grid:Grid, start:tuple, goal:tuple):
    """
    A* with admissible heuristic: Manhattan distance * min_cell_cost
    """
    t0 = time.perf_counter()
    min_cost = min(grid.cells[r][c] for r in range(grid.R) for c in range(grid.C) if grid.cells[r][c] is not None)
    def h(a,b):
        (r1,c1)=a; (r2,c2)=b
        return (abs(r1-r2)+abs(c1-c2))*min_cost

    open_pq = []
    heapq.heappush(open_pq, (h(start,goal), 0, start))
    came_from = {}
    gscore = {start: 0}
    nodes_expanded = 0
    while open_pq:
        f, g, curr = heapq.heappop(open_pq)
        nodes_expanded += 1
        if curr == goal:
            elapsed = time.perf_counter() - t0
            path = reconstruct_path(came_from, goal)
            return {'path': path, 'cost': g, 'expanded': nodes_expanded, 'time': elapsed}
        for nbr in grid.neighbors4(*curr):
            nr,nc = nbr
            tentative_g = g + grid.cost(nr, nc)
            if nbr not in gscore or tentative_g < gscore[nbr]:
                gscore[nbr] = tentative_g
                fscore = tentative_g + h(nbr, goal)
                heapq.heappush(open_pq, (fscore, tentative_g, nbr))
                came_from[nbr] = curr
    return None

# -------------------------
# dynamic replanning helper
# -------------------------
def simulate_with_dynamic_replanning(grid:Grid, start:tuple, goal:tuple, schedule:dict, planner_fn, time_limit_steps:int=10000):
    """
    Planner_fn: function(grid, start, goal) -> same result dict as other planners.
    Agent moves 1 step per time-step. Before moving into the next cell it checks schedule at t+1.
    If blocked at t+1, the agent waits (or you may choose to replan immediately depending on algorithm).
    Returns metrics: time_steps, log (list of tuples), nodes_expanded, sim_time, total_cost
    """
    t = 0
    current = start
    log = []
    total_nodes = 0
    total_planner_time = 0.0
    visited_positions = [start]

    while current != goal:
        res = planner_fn(grid, current, goal)
        if not res:
            log.append((t, current, 'NO_PATH'))
            break
        total_nodes += res['expanded']
        total_planner_time += res['time']
        path = res['path']
        if len(path) < 2:
            log.append((t, current, 'AT_GOAL_OR_STUCK'))
            break
        nextcell = path[1]
        # check schedule for occupation at time t+1
        if (t+1) in schedule and nextcell in schedule[t+1]:
            log.append((t, current, f'BLOCKED_NEXT {nextcell} at t+1; WAIT'))
            t += 1
            if t > time_limit_steps:
                log.append((t, current, 'TIMEOUT'))
                break
            continue
        # move
        log.append((t, current, f'MOVE_TO {nextcell}'))
        current = nextcell
        visited_positions.append(current)
        t += 1
        if t > time_limit_steps:
            log.append((t, current, 'TIMEOUT'))
            break

    total_cost = None
    if current == goal:
        total_cost = sum(grid.cost(r,c) for (r,c) in visited_positions[1:])
    return {'time_steps': t, 'log': log, 'nodes_expanded': total_nodes, 'sim_time': total_planner_time, 'total_cost': total_cost}
