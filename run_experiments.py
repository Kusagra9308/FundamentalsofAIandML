# run_experiments.py
"""
Simple experiment harness:
 - Runs planners on a list of maps and start/goal pairs
 - Writes results CSV (map, planner, path_cost, nodes_expanded, time_s, path_len)
 - Optionally makes a small plot (requires matplotlib)
Usage:
  python run_experiments.py --maps maps/small.map maps/dynamic.map --planners astar ucs bfs local_hill local_sa --start 0,0 --goal 9,9 --out results/results_all.csv
"""

import argparse
import csv
import os
import search
import planner as cli
import matplotlib.pyplot as plt

def run_once(mapfile, planner_name, start, goal):
    grid = search.parse_map_file(mapfile)
    sr, sc = start; gr, gc = goal
    if not grid.passable(sr, sc) or not grid.passable(gr, gc):
        return None
    if planner_name in ['local_hill','local_sa']:
        res = cli.run_planner_by_name(planner_name, grid, start, goal)
        # local search functions return {'path':..., 'cost':...}
        if not res:
            return None
        return {'path_cost': res['cost'], 'nodes_expanded': None, 'time': None, 'path_len': len(res['path'])}
    else:
        res = cli.run_planner_by_name(planner_name, grid, start, goal)
        if not res:
            return None
        return {'path_cost': res['cost'], 'nodes_expanded': res.get('expanded'), 'time': res.get('time'), 'path_len': len(res['path'])}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps', nargs='+', required=True)
    parser.add_argument('--planners', nargs='+', required=True)
    parser.add_argument('--start', default='0,0')
    parser.add_argument('--goal', default='9,9')
    parser.add_argument('--out', default='results/results_all.csv')
    args = parser.parse_args()

    start = tuple(map(int, args.start.split(',')))
    goal = tuple(map(int, args.goal.split(',')))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    rows = []
    for m in args.maps:
        for p in args.planners:
            print("Running", p, "on", m)
            r = run_once(m, p, start, goal)
            if r is None:
                print("No result for", p, m)
                rows.append({'map':m,'planner':p,'path_cost':None,'nodes_expanded':None,'time_s':None,'path_len':None})
            else:
                rows.append({'map':m,'planner':p,'path_cost':r['path_cost'],'nodes_expanded':r['nodes_expanded'],'time_s':r['time'],'path_len':r['path_len']})

    # write CSV
    with open(args.out,'w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['map','planner','path_cost','nodes_expanded','time_s','path_len'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("Wrote results to", args.out)

    # small plot (mean path_cost per planner if numeric)
    try:
        import pandas as pd
        df = pd.read_csv(args.out)
        pivot = df.pivot_table(index='planner', values='path_cost', aggfunc='mean')
        pivot = pivot.dropna()
        pivot.plot.bar(y='path_cost', legend=False, title='Mean path cost by planner')
        plt.tight_layout()
        plt.savefig(os.path.splitext(args.out)[0] + '_plot.png')
        print("Saved plot.")
    except Exception as e:
        print("Skipping plot (pandas/matplotlib missing or error):", e)

if __name__ == '__main__':
    main()
