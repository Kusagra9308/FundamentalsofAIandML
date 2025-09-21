# planner.py
"""
Command-line interface for running planners and dynamic simulation.
Usage examples:
  python planner.py --map maps/small.map --planner astar --start 0,0 --goal 4,4
  python planner.py --map maps/dynamic.map --dyn_schedule maps/dynamic.schedule --planner astar --start 0,0 --goal 9,9 --simulate_dynamic
  python planner.py --map maps/medium.map --planner local_hill --start 0,0 --goal 14,14
"""

import argparse
import sys
import search
import local_search
import json

def run_planner_by_name(name, grid, start, goal):
    if name == 'bfs':
        return search.bfs(grid, start, goal)
    if name == 'ucs':
        return search.ucs(grid, start, goal)
    if name == 'astar':
        return search.astar(grid, start, goal)
    if name == 'local_hill':
        return local_search.hill_climbing_random_restarts(grid, start, goal)
    if name == 'local_sa':
        return local_search.simulated_annealing_repair(grid, start, goal)
    raise ValueError("Unknown planner")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', required=True, help='map file')
    parser.add_argument('--dyn_schedule', default=None, help='dynamic schedule file')
    parser.add_argument('--planner', default='astar', choices=['bfs','ucs','astar','local_hill','local_sa'])
    parser.add_argument('--start', default='0,0', help='start r,c')
    parser.add_argument('--goal', default='0,0', help='goal r,c')
    parser.add_argument('--simulate_dynamic', action='store_true', help='run dynamic simulation (requires schedule)')
    parser.add_argument('--out_json', default=None, help='write JSON of result metrics')
    args = parser.parse_args()

    grid = search.parse_map_file(args.map)
    sr, sc = map(int, args.start.split(','))
    gr, gc = map(int, args.goal.split(','))
    start = (sr, sc); goal = (gr, gc)

    if not grid.passable(*start) or not grid.passable(*goal):
        print("Start or goal blocked. Exiting.")
        sys.exit(1)

    if args.simulate_dynamic:
        if not args.dyn_schedule:
            print("Dynamic simulation needs --dyn_schedule")
            sys.exit(1)
        schedule = search.load_schedule(args.dyn_schedule)
        metrics = search.simulate_with_dynamic_replanning(grid, start, goal, schedule, lambda g,s,g2: run_planner_by_name(args.planner, g, s, g2))
        print("Simulation metrics:")
        print("time_steps:", metrics['time_steps'])
        print("nodes_expanded total:", metrics['nodes_expanded'])
        print("sim_time (sum of planner times):", metrics['sim_time'])
        print("total_cost:", metrics['total_cost'])
        print("log entries (first 50):")
        for e in metrics['log'][:50]:
            print(e)
        if args.out_json:
            with open(args.out_json,'w') as f:
                json.dump(metrics, f, indent=2)
    else:
        res = run_planner_by_name(args.planner, grid, start, goal)
        if not res:
            print("No path found.")
            sys.exit(1)
        print("Planner:", args.planner)
        print("Path length:", len(res['path']))
        print("Path cost:", res['cost'])
        print("Nodes expanded:", res.get('expanded'))
        print("Time (s):", res.get('time'))
        if args.out_json:
            with open(args.out_json,'w') as f:
                json.dump(res, f, indent=2)

if __name__ == '__main__':
    main()
